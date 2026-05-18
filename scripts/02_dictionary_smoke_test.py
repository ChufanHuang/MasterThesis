#!/usr/bin/env python3
"""
Run a smoke test for dictionary-based narrative matching.

Inputs:
    data/clean/clean_conspiracy_en.csv
    data/dictionaries/science_related.yaml

Outputs:
    data/processed/labeled_sample.csv
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_PATH = PROJECT_ROOT / "data" / "clean" / "clean_conspiracy_en.csv"
DICTIONARY_PATH = PROJECT_ROOT / "data" / "dictionaries" / "science_related.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "labeled_sample.csv"

SAMPLE_SIZE = 1_000
INSPECTION_SIZE = 50
RANDOM_SEED = 42


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

    for column in ["text", "text_norm", "hashtags"]:
        if column not in df.columns:
            df[column] = ""

        df[column] = df[column].fillna("").astype(str)

    return df


def keyword_matches(text: str, keyword: str) -> bool:
    if not text or not keyword:
        return False

    pattern = rf"\b{re.escape(keyword)}\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def label_row(
    text_norm: str,
    hashtags: str,
    dictionary: dict[str, Any],
) -> list[dict[str, Any]]:
    matches = []

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
            matches.append(
                {
                    "narrative": narrative,
                    "is_science_related": bool(
                        config.get("is_science_related", False)
                    ),
                }
            )

    return matches


def sample_data(df: pd.DataFrame) -> pd.DataFrame:
    sample_size = min(SAMPLE_SIZE, len(df))
    return df.sample(n=sample_size, random_state=RANDOM_SEED).copy()


def add_dictionary_labels(
    df: pd.DataFrame,
    dictionary: dict[str, Any],
) -> pd.DataFrame:
    df = df.copy()

    df["matched_narratives"] = df.apply(
        lambda row: label_row(
            row["text_norm"],
            row["hashtags"],
            dictionary,
        ),
        axis=1,
    )

    df["matched_narrative_count"] = df["matched_narratives"].apply(len)
    df["has_science_related_match"] = df["matched_narratives"].apply(
        lambda matches: any(match["is_science_related"] for match in matches)
    )

    return df


def print_match_summary(df: pd.DataFrame) -> None:
    exploded = df["matched_narratives"].explode().dropna()
    total_labels = len(exploded)

    if total_labels > 0:
        science_labels = int(
            exploded.apply(
                lambda item: bool(item.get("is_science_related", False))
            ).sum()
        )
    else:
        science_labels = 0

    print("\nDictionary smoke test")
    print(f"Sample size:            {len(df):,}")
    print(f"Matched labels:         {total_labels:,}")
    print(f"Science-related labels: {science_labels:,}")

    if total_labels > 0:
        print(f"Science-related share:  {science_labels / total_labels:.2%}")
    else:
        print("Science-related share:  0.00%")


def print_manual_inspection_examples(df: pd.DataFrame) -> None:
    random.seed(RANDOM_SEED)

    inspection_size = min(INSPECTION_SIZE, len(df))
    selected_positions = random.sample(range(len(df)), inspection_size)
    inspection_df = df.iloc[selected_positions].copy()

    print("\nManual inspection examples")
    print("-" * 100)

    for _, row in inspection_df.head(20).iterrows():
        preview = str(row["text"])[:120].replace("\n", " ")
        science_narratives = [
            match["narrative"]
            for match in row["matched_narratives"]
            if match.get("is_science_related")
        ]

        print(f"Tweet preview: {preview}")
        print(
            "Science narratives: "
            f"{science_narratives if science_narratives else 'None'}"
        )
        print("-" * 100)


def main() -> None:
    ensure_output_dirs()

    df = load_clean_data(CLEAN_PATH)
    dictionary = load_dictionary(DICTIONARY_PATH)

    df = prepare_text_fields(df)
    sample_df = sample_data(df)
    labeled_sample = add_dictionary_labels(sample_df, dictionary)

    print_match_summary(labeled_sample)
    print_manual_inspection_examples(labeled_sample)

    labeled_sample.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\nLabeled sample saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()