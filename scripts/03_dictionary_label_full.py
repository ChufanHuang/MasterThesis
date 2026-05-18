#!/usr/bin/env python3
"""

Thesis section:
   4.1 Dictionary Annotation Results
     4.1.1 Overall Annotation Results
     4.1.2 Overall Distribution of Science-Related Conspiracy Narratives


Inputs:
    data/clean/clean_conspiracy_en.csv
    data/dictionaries/science_related.yaml

Outputs:
    data/processed/labeled_all.csv
    data/processed/science_conspiracy_tweets.csv
    results/03_dictionary_label_full/labeling_summary.csv
    results/03_dictionary_label_full/science_narrative_distribution.csv
    figures/03_dictionary_label_full/science_narrative_distribution.png
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_PATH = PROJECT_ROOT / "data" / "clean" / "clean_conspiracy_en.csv"
DICTIONARY_PATH = PROJECT_ROOT / "data" / "dictionaries" / "science_related.yaml"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "03_dictionary_label_full"
FIGURE_DIR = PROJECT_ROOT / "figures" / "03_dictionary_label_full"

LABELED_ALL_PATH = PROCESSED_DIR / "labeled_all.csv"
SCIENCE_TWEETS_PATH = PROCESSED_DIR / "science_conspiracy_tweets.csv"
SUMMARY_PATH = RESULTS_DIR / "labeling_summary.csv"
DISTRIBUTION_TABLE_PATH = RESULTS_DIR / "science_narrative_distribution.csv"
DISTRIBUTION_FIGURE_PATH = FIGURE_DIR / "science_narrative_distribution.png"

TOP_N_FOR_PLOT = 10
HIGHLIGHT_LABEL = "Flat Earth"
FIGURE_SIZE = (10, 6)


def ensure_output_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_clean_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned data file not found: {path}. "
            "Run scripts/01_clean_data.py first."
        )

    return pd.read_csv(path, low_memory=False)


def load_dictionary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dictionary file not found: {path}. "
            "Place science_related.yaml under data/dictionaries/."
        )

    with path.open("r", encoding="utf-8") as file:
        dictionary = yaml.safe_load(file)

    if not isinstance(dictionary, dict):
        raise ValueError("Dictionary file must contain a YAML mapping.")

    return dictionary


def prepare_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in ["text_norm", "hashtags"]:
        if column not in df.columns:
            df[column] = ""

        df[column] = df[column].fillna("").astype(str)

    return df


def keyword_matches(text: str, keyword: str) -> bool:
    if not text or not keyword:
        return False

    pattern = rf"\b{re.escape(str(keyword))}\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def label_row(
    text_norm: str,
    hashtags: str,
    dictionary: dict[str, Any],
) -> tuple[list[str], list[str], bool]:
    all_narratives = []
    science_narratives = []

    for narrative, config in dictionary.items():
        if not isinstance(config, dict):
            continue

        keywords = config.get("keywords_core", [])
        if not keywords:
            continue

        has_match = any(
            keyword_matches(text_norm, keyword) or keyword_matches(hashtags, keyword)
            for keyword in keywords
        )

        if has_match:
            all_narratives.append(narrative)

            if bool(config.get("is_science_related", False)):
                science_narratives.append(narrative)

    return all_narratives, science_narratives, bool(science_narratives)


def apply_dictionary_labels(
    df: pd.DataFrame,
    dictionary: dict[str, Any],
) -> pd.DataFrame:
    df = df.copy()

    tqdm.pandas(desc="Dictionary labeling")
    labels = df.progress_apply(
        lambda row: label_row(
            row["text_norm"],
            row["hashtags"],
            dictionary,
        ),
        axis=1,
    )

    df["all_narratives"] = labels.apply(lambda item: item[0])
    df["science_narratives"] = labels.apply(lambda item: item[1])
    df["is_science_related"] = labels.apply(lambda item: item[2])
    df["all_narrative_count"] = df["all_narratives"].apply(len)
    df["science_narrative_count"] = df["science_narratives"].apply(len)

    return df


def build_labeling_summary(df: pd.DataFrame, df_science: pd.DataFrame) -> pd.DataFrame:
    total_tweets = len(df)
    science_tweets = len(df_science)

    summary = pd.DataFrame(
        [
            {
                "metric": "total_tweets",
                "value": total_tweets,
            },
            {
                "metric": "science_related_tweets",
                "value": science_tweets,
            },
            {
                "metric": "share_science_related_percent",
                "value": round(science_tweets / total_tweets * 100, 2)
                if total_tweets
                else 0.0,
            },
        ]
    )

    return summary


def build_distribution_table(
    df_all: pd.DataFrame,
    df_science: pd.DataFrame,
) -> pd.DataFrame:
    counts = df_science["science_narratives"].explode().value_counts(dropna=True)

    science_tweet_count = len(df_science)
    full_tweet_count = len(df_all)

    table = (
        counts.rename_axis("narrative")
        .reset_index(name="labeled_tweets")
        .sort_values("labeled_tweets", ascending=False)
        .reset_index(drop=True)
    )

    table["share_within_science_percent"] = (
        table["labeled_tweets"] / science_tweet_count * 100
    ).round(2)

    table["cumulative_share_percent"] = (
        table["share_within_science_percent"].cumsum()
    ).round(2)

    table["share_within_full_percent"] = (
        table["labeled_tweets"] / full_tweet_count * 100
    ).round(3)

    table["science_tweet_count"] = science_tweet_count
    table["full_tweet_count"] = full_tweet_count

    return table


def prepare_plot_table(table: pd.DataFrame) -> pd.DataFrame:
    plot_table = table.copy()

    if TOP_N_FOR_PLOT is not None and len(plot_table) > TOP_N_FOR_PLOT:
        top_rows = plot_table.iloc[:TOP_N_FOR_PLOT].copy()
        remaining_rows = plot_table.iloc[TOP_N_FOR_PLOT:].copy()

        other_row = pd.DataFrame(
            [
                {
                    "narrative": "Other",
                    "labeled_tweets": int(remaining_rows["labeled_tweets"].sum()),
                    "share_within_science_percent": round(
                        float(remaining_rows["share_within_science_percent"].sum()),
                        2,
                    ),
                }
            ]
        )

        plot_table = pd.concat([top_rows, other_row], ignore_index=True)

    return plot_table.sort_values("share_within_science_percent", ascending=True)


def save_distribution_figure(table: pd.DataFrame, output_path: Path) -> None:
    plot_table = prepare_plot_table(table)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.barh(
        plot_table["narrative"],
        plot_table["share_within_science_percent"],
        edgecolor="black",
        linewidth=0.6,
    )

    if HIGHLIGHT_LABEL and HIGHLIGHT_LABEL in set(plot_table["narrative"]):
        row = plot_table[plot_table["narrative"] == HIGHLIGHT_LABEL].iloc[0]
        ax.barh(
            [row["narrative"]],
            [row["share_within_science_percent"]],
            edgecolor="black",
            linewidth=1.4,
            hatch="///",
            fill=False,
        )

    ax.set_xlabel("Share within science-related tweets (%)")
    ax.set_ylabel("Narrative label")
    ax.set_title(
        "Distribution of Science-Related Conspiracy Narrative Labels",
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for index, value in enumerate(plot_table["share_within_science_percent"]):
        ax.text(value + 0.3, index, f"{value:.2f}%", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary: pd.DataFrame, distribution_table: pd.DataFrame) -> None:
    print("\nFull-corpus labeling summary")
    print(summary.to_string(index=False))

    print("\nScience narrative distribution")
    print(
        distribution_table[
            [
                "narrative",
                "labeled_tweets",
                "share_within_science_percent",
                "cumulative_share_percent",
                "share_within_full_percent",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


def main() -> None:
    ensure_output_dirs()

    df = load_clean_data(CLEAN_PATH)
    dictionary = load_dictionary(DICTIONARY_PATH)

    df = prepare_text_fields(df)

    print(f"Loaded cleaned corpus: {len(df):,} tweets")
    print(f"Loaded dictionary narratives: {len(dictionary):,}")

    labeled_df = apply_dictionary_labels(df, dictionary)
    science_df = labeled_df[labeled_df["is_science_related"]].copy()

    summary = build_labeling_summary(labeled_df, science_df)
    distribution_table = build_distribution_table(labeled_df, science_df)

    labeled_df.to_csv(LABELED_ALL_PATH, index=False, encoding="utf-8")
    science_df.to_csv(SCIENCE_TWEETS_PATH, index=False, encoding="utf-8")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8")
    distribution_table.to_csv(DISTRIBUTION_TABLE_PATH, index=False, encoding="utf-8")

    save_distribution_figure(distribution_table, DISTRIBUTION_FIGURE_PATH)

    print_summary(summary, distribution_table)

    print("\nSaved outputs")
    print(f"Labeled corpus:       {LABELED_ALL_PATH}")
    print(f"Science subcorpus:    {SCIENCE_TWEETS_PATH}")
    print(f"Labeling summary:     {SUMMARY_PATH}")
    print(f"Distribution table:   {DISTRIBUTION_TABLE_PATH}")
    print(f"Distribution figure:  {DISTRIBUTION_FIGURE_PATH}")


if __name__ == "__main__":
    main()