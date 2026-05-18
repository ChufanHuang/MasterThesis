#!/usr/bin/env python3
"""
Compute sentiment baseline characteristics for science-related narratives.

Thesis section:
    4.4.1 Sentiment Baseline Characteristics

Inputs:
    data/processed/science_conspiracy_tweets.csv

Outputs:
    results/10_sentiment_baseline/sentiment_tweet_level.csv
    results/10_sentiment_baseline/sentiment_monthly_agg.csv
    results/10_sentiment_baseline/sentiment_baseline_by_narrative.csv

Notes:
    This script uses VADER sentiment with a small domain lexicon update.
    Negative: compound < -0.2
    Neutral:  -0.2 <= compound <= 0.2
    Positive: compound > 0.2

    The analysis uses single-label tweets only, consistent with the framing
    pipeline.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "science_conspiracy_tweets.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "10_sentiment_baseline"

TWEET_LEVEL_PATH = RESULTS_DIR / "sentiment_tweet_level.csv"
MONTHLY_AGG_PATH = RESULTS_DIR / "sentiment_monthly_agg.csv"
BASELINE_PATH = RESULTS_DIR / "sentiment_baseline_by_narrative.csv"

TEXT_COLUMN_PRIMARY = "text_norm"
TEXT_COLUMN_FALLBACK = "text"
TIME_COLUMN = "created_at"
LABEL_COLUMN = "science_narratives"

NEGATIVE_THRESHOLD = -0.2
POSITIVE_THRESHOLD = 0.2

CUSTOM_LEXICON = {
    "hoax": -3.0,
    "lie": -3.0,
    "coverup": -2.5,
    "cover-up": -2.5,
    "poison": -2.5,
    "danger": -2.0,
    "deceive": -2.0,
    "truth": 1.5,
    "expose": 1.0,
}


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def initialize_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    nltk.download("vader_lexicon", quiet=True)
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(CUSTOM_LEXICON)

    return analyzer


def parse_list_cell(value: object) -> list[str]:
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [
                    str(item).strip()
                    for item in parsed
                    if str(item).strip()
                ]
        except (ValueError, SyntaxError):
            pass

    return [text]


def polarity_from_valence(valence: float) -> str:
    if valence < NEGATIVE_THRESHOLD:
        return "Negative"

    if valence > POSITIVE_THRESHOLD:
        return "Positive"

    return "Neutral"


def compute_sentiment(
    text: str,
    analyzer: SentimentIntensityAnalyzer,
) -> tuple[float, float, str]:
    scores = analyzer.polarity_scores(text)

    valence = float(scores["compound"])
    intensity = float(scores["pos"] + scores["neg"])
    polarity = polarity_from_valence(valence)

    return valence, intensity, polarity


def pick_id_column(df: pd.DataFrame) -> str | None:
    for column in ["status_id", "tweet_id", "id"]:
        if column in df.columns:
            return column

    return None


def load_science_tweets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. "
            "Run scripts/03_dictionary_label_full.py first."
        )

    df = pd.read_csv(path, low_memory=False)

    required_columns = {TIME_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if TEXT_COLUMN_PRIMARY not in df.columns and TEXT_COLUMN_FALLBACK not in df.columns:
        raise ValueError(
            f"Missing text columns: {TEXT_COLUMN_PRIMARY} or {TEXT_COLUMN_FALLBACK}"
        )

    return df


def prepare_single_label_tweets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(parse_list_cell)

    df = df[
        df[LABEL_COLUMN].apply(lambda labels: isinstance(labels, list) and len(labels) == 1)
    ].copy()

    df["narrative"] = df[LABEL_COLUMN].apply(lambda labels: labels[0])

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce", utc=True)
    df = df.dropna(subset=[TIME_COLUMN]).copy()
    df[TIME_COLUMN] = df[TIME_COLUMN].dt.tz_convert(None)

    text_column = (
        TEXT_COLUMN_PRIMARY
        if TEXT_COLUMN_PRIMARY in df.columns
        else TEXT_COLUMN_FALLBACK
    )

    df["text_to_analyze"] = df[text_column].fillna("").astype(str).str.strip()
    df = df[df["text_to_analyze"].str.len() > 0].copy()

    df["year_month"] = df[TIME_COLUMN].dt.strftime("%Y-%m")

    return df


def add_sentiment_columns(
    df: pd.DataFrame,
    analyzer: SentimentIntensityAnalyzer,
) -> pd.DataFrame:
    df = df.copy()

    sentiment_values = df["text_to_analyze"].apply(
        lambda text: compute_sentiment(text, analyzer)
    )

    df["valence"] = sentiment_values.apply(lambda item: item[0])
    df["intensity"] = sentiment_values.apply(lambda item: item[1])
    df["polarity"] = sentiment_values.apply(lambda item: item[2])

    return df


def build_tweet_level_output(df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = []

    id_column = pick_id_column(df)
    if id_column is not None:
        keep_columns.append(id_column)

    if "frame" in df.columns:
        keep_columns.append("frame")

    keep_columns += [
        "narrative",
        TIME_COLUMN,
        "year_month",
        "valence",
        "intensity",
        "polarity",
        "text_to_analyze",
    ]

    return df[keep_columns].copy()


def average_negative_intensity(group: pd.DataFrame) -> float:
    negative_rows = group[group["polarity"] == "Negative"]

    if negative_rows.empty:
        return 0.0

    return float(negative_rows["intensity"].mean())


def build_monthly_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby(["narrative", "year_month"], as_index=False)
        .agg(
            tweet_count=("text_to_analyze", "count"),
            avg_valence=("valence", "mean"),
            avg_intensity=("intensity", "mean"),
            negative_ratio_percent=("polarity", lambda x: (x == "Negative").mean() * 100),
            neutral_ratio_percent=("polarity", lambda x: (x == "Neutral").mean() * 100),
            positive_ratio_percent=("polarity", lambda x: (x == "Positive").mean() * 100),
        )
    )

    negative_intensity = (
        df.groupby(["narrative", "year_month"])
        .apply(average_negative_intensity)
        .reset_index(name="avg_negative_intensity")
    )

    monthly = monthly.merge(
        negative_intensity,
        on=["narrative", "year_month"],
        how="left",
    )

    monthly["year_month_sort"] = pd.to_datetime(
        monthly["year_month"],
        format="%Y-%m",
        errors="coerce",
    )

    monthly = (
        monthly.sort_values(["narrative", "year_month_sort"])
        .drop(columns=["year_month_sort"])
        .round(3)
    )

    return monthly


def build_baseline_by_narrative(df: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        df.groupby("narrative", as_index=False)
        .agg(
            total_tweet_count=("text_to_analyze", "count"),
            avg_valence=("valence", "mean"),
            avg_intensity=("intensity", "mean"),
            negative_ratio_percent=("polarity", lambda x: (x == "Negative").mean() * 100),
            neutral_ratio_percent=("polarity", lambda x: (x == "Neutral").mean() * 100),
            positive_ratio_percent=("polarity", lambda x: (x == "Positive").mean() * 100),
        )
    )

    negative_intensity = (
        df.groupby("narrative")
        .apply(average_negative_intensity)
        .reset_index(name="avg_negative_intensity")
    )

    baseline = baseline.merge(negative_intensity, on="narrative", how="left")
    baseline = baseline.round(3)

    return baseline.sort_values(
        "total_tweet_count",
        ascending=False,
    ).reset_index(drop=True)


def main() -> None:
    ensure_output_dirs()

    analyzer = initialize_sentiment_analyzer()

    df = load_science_tweets(INPUT_PATH)
    df = prepare_single_label_tweets(df)
    df = add_sentiment_columns(df, analyzer)

    tweet_level = build_tweet_level_output(df)
    monthly_agg = build_monthly_aggregation(df)
    baseline = build_baseline_by_narrative(df)

    tweet_level.to_csv(TWEET_LEVEL_PATH, index=False, encoding="utf-8")
    monthly_agg.to_csv(MONTHLY_AGG_PATH, index=False, encoding="utf-8")
    baseline.to_csv(BASELINE_PATH, index=False, encoding="utf-8")

    print("\nSaved outputs")
    print(f"Tweet-level sentiment: {TWEET_LEVEL_PATH}")
    print(f"Monthly aggregation:   {MONTHLY_AGG_PATH}")
    print(f"Baseline table:        {BASELINE_PATH}")

    print("\nSummary")
    print(f"Tweets after filtering: {len(df):,}")
    print(f"Time range:             {df[TIME_COLUMN].min()} to {df[TIME_COLUMN].max()}")

    print("\nBaseline preview")
    print(baseline.head(10).to_string(index=False))


if __name__ == "__main__":
    main()