#!/usr/bin/env python3
"""
Clean the raw conspiracy tweet dataset.

Inputs:
    data/raw/conspiracy_tweets.csv

Outputs:
    data/clean/clean_conspiracy_en.csv
    figures/01_clean_data/language_distribution_top15.png
    figures/01_clean_data/yearly_compare.png
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "conspiracy_tweets.csv"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
FIGURE_DIR = PROJECT_ROOT / "figures" / "01_clean_data"

OUTPUT_PATH = CLEAN_DIR / "clean_conspiracy_en.csv"

CORE_COLUMNS = {
    "created_at",
    "status_id",
    "user_id",
    "screen_name",
    "text",
    "hashtags",
    "lang",
    "is_retweet",
    "retweet_count",
    "favorite_count",
    "in_reply_to_status_id",
    "in_reply_to_status_id_str",
    "quoted_status_id",
    "quoted_status_id_str",
    "retweeted_status_id",
    "retweeted_status_id_str",
    "user_screen_name",
    "username",
}

MENTION_PATTERN = re.compile(r"@(\w+)")


def ensure_output_dirs() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    return pd.read_csv(path, low_memory=False)


def select_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in df.columns if column in CORE_COLUMNS]
    selected = df[columns].copy()

    print(
        f"Loaded {len(df):,} rows and kept "
        f"{selected.shape[1]} of {df.shape[1]} columns."
    )

    return selected


def normalize_text(value: object) -> str:
    text = str(value or "")
    text = text.replace("\u2019", "'").replace("&amp;", "&")
    text = re.sub(r"\b9[\s/\\-]*11\b", "911", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def extract_mentions(value: object) -> str:
    text = str(value or "")
    return " ".join(MENTION_PATTERN.findall(text))


def normalize_hashtags(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            tags = [str(item).strip().lstrip("#").lower() for item in parsed]
            return " ".join(f"#{tag}" for tag in tags if tag)
        except (ValueError, SyntaxError):
            pass

    parts = re.split(r"[,|\s;]+", text)
    tags = [part.strip().lstrip("#").lower() for part in parts if part.strip()]

    return " ".join(f"#{tag}" for tag in tags)


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["year"] = df["created_at"].dt.year.astype("Int16")
    df["month"] = df["created_at"].dt.month.astype("Int8")

    return df


def add_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "text" not in df.columns:
        df["text"] = ""

    df["mentions"] = df["text"].apply(extract_mentions)
    df["text_norm"] = df["text"].apply(normalize_text)

    return df


def save_language_distribution_plot(
    df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> None:
    if "lang" not in df.columns:
        print("Skipping language distribution plot because 'lang' is missing.")
        return

    language_counts = df["lang"].value_counts(dropna=False)
    top_languages = language_counts.head(top_n)
    other_count = int(language_counts.iloc[top_n:].sum())

    if other_count > 0:
        plot_data = pd.concat([top_languages, pd.Series({"other": other_count})])
    else:
        plot_data = top_languages

    plt.figure(figsize=(7, 4))
    plot_data.plot(kind="bar")
    plt.ylabel("Tweets")
    plt.title(f"Top {top_n} language tags")

    for index, value in enumerate(plot_data.values):
        label_offset = max(10, value * 0.01)
        plt.text(index, value + label_offset, f"{value:,}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_yearly_comparison_plot(
    df_all: pd.DataFrame,
    df_english: pd.DataFrame,
    output_path: Path,
) -> None:
    def yearly_counts(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby(frame["created_at"].dt.year).size().sort_index()

    plt.figure(figsize=(8, 4))
    yearly_counts(df_all).plot(label="All languages", linewidth=2)
    yearly_counts(df_english).plot(label="English only", linewidth=2)

    plt.title("Yearly tweet counts: English vs. all languages")
    plt.ylabel("Tweets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def filter_english_tweets(df: pd.DataFrame) -> pd.DataFrame:
    if "lang" not in df.columns:
        print("Column 'lang' is missing. Keeping all rows.")
        return df.copy()

    filtered = df[df["lang"] == "en"].copy()
    print(f"Language filter: {len(df):,} -> {len(filtered):,} rows")

    return filtered


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if "status_id" not in df.columns:
        print("Column 'status_id' is missing. Skipping deduplication.")
        return df.copy()

    duplicate_count = int(df.duplicated("status_id").sum())
    deduplicated = df.drop_duplicates("status_id").copy()

    print(f"Duplicate check: removed {duplicate_count:,} duplicate rows")

    return deduplicated


def finalize_hashtags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "hashtags" not in df.columns:
        df["hashtags"] = ""
        print("Column 'hashtags' is missing. Added empty hashtag field.")
        return df

    missing_count = int(df["hashtags"].isna().sum())
    df["hashtags"] = df["hashtags"].apply(normalize_hashtags)

    print(f"Hashtags normalized. Missing values filled: {missing_count:,}")

    return df


def print_summary(df: pd.DataFrame) -> None:
    unique_users = df["user_id"].nunique() if "user_id" in df.columns else None

    print("\nSummary")
    print(f"Rows:         {len(df):,}")
    print(f"Unique users: {unique_users}")
    print(f"Start date:   {df['created_at'].min()}")
    print(f"End date:     {df['created_at'].max()}")


def main() -> None:
    ensure_output_dirs()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw input:    {RAW_PATH}")

    raw_df = load_raw_data(RAW_PATH)
    df = select_core_columns(raw_df)
    df = add_time_columns(df)
    df = add_text_columns(df)

    save_language_distribution_plot(
        df,
        FIGURE_DIR / "language_distribution_top15.png",
    )

    english_df = filter_english_tweets(df)

    save_yearly_comparison_plot(
        df,
        english_df,
        FIGURE_DIR / "yearly_compare.png",
    )

    english_df = remove_duplicates(english_df)
    english_df = finalize_hashtags(english_df)

    english_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\nCleaned data saved to: {OUTPUT_PATH}")

    print_summary(english_df)


if __name__ == "__main__":
    main()