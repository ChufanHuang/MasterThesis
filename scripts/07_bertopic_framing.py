#!/usr/bin/env python3
"""
Run BERTopic framing within each science-related narrative.

Thesis section:
    4.3.1 Internal Frame Structure of Each Narrative

Inputs:
    data/processed/science_conspiracy_tweets.csv

Outputs:
    models/07_bertopic_framing/{narrative}.pkl
    results/07_bertopic_framing/{narrative}_frame_info.csv
    results/07_bertopic_framing/{narrative}_tweets_with_frames.csv
    figures/07_bertopic_framing/{narrative}_topics.html

Notes:
    This script uses single-label tweets only.
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "science_conspiracy_tweets.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "07_bertopic_framing"
RESULTS_DIR = PROJECT_ROOT / "results" / "07_bertopic_framing"
FIGURE_DIR = PROJECT_ROOT / "figures" / "07_bertopic_framing"

TEXT_COLUMN = "text_norm"
TIME_COLUMN = "created_at"
LABEL_COLUMN = "science_narratives"

MIN_DOCS_TO_MODEL = 10
MIN_TOPIC_SIZE = 2

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 42

NGRAM_RANGE = (1, 2)
STOP_WORDS = "english"


def ensure_output_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_science_tweets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. "
            "Run scripts/03_dictionary_label_full.py first."
        )

    df = pd.read_csv(path, low_memory=False)

    required_columns = {TEXT_COLUMN, TIME_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return df


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


def slugify(value: str) -> str:
    slug = value.strip()
    slug = slug.replace("/", "_")
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "", slug)

    return slug or "unknown"


def prepare_single_label_tweets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(parse_list_cell)

    single_label_df = df[
        df[LABEL_COLUMN].apply(lambda labels: isinstance(labels, list) and len(labels) == 1)
    ].copy()

    single_label_df["narrative"] = single_label_df[LABEL_COLUMN].apply(
        lambda labels: labels[0]
    )

    single_label_df[TEXT_COLUMN] = (
        single_label_df[TEXT_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    single_label_df = single_label_df[single_label_df[TEXT_COLUMN].str.len() > 0].copy()

    return single_label_df


def get_adaptive_umap_parameters(n_docs: int) -> dict:
    n_neighbors = int(np.clip(round(np.sqrt(n_docs)), 5, 30))
    n_neighbors = min(n_neighbors, max(5, n_docs - 1))

    return {
        "n_neighbors": n_neighbors,
        "n_components": 5,
        "min_dist": 0.0,
        "metric": "cosine",
        "random_state": RANDOM_STATE,
    }


def get_adaptive_hdbscan_parameters(n_docs: int) -> dict:
    min_cluster_size = int(np.clip(round(n_docs * 0.02), 2, 20))
    min_samples = max(1, min_cluster_size // 2)

    return {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "prediction_data": False,
    }


def build_topic_model(n_docs: int) -> BERTopic:
    vectorizer_model = CountVectorizer(
        ngram_range=NGRAM_RANGE,
        stop_words=STOP_WORDS,
    )

    representation_model = KeyBERTInspired()
    umap_model = UMAP(**get_adaptive_umap_parameters(n_docs))
    hdbscan_model = hdbscan.HDBSCAN(**get_adaptive_hdbscan_parameters(n_docs))

    return BERTopic(
        embedding_model=EMBEDDING_MODEL,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        language="english",
        min_topic_size=MIN_TOPIC_SIZE,
        calculate_probabilities=False,
        verbose=True,
    )


def write_skip_note(narrative: str, reason: str) -> None:
    output_path = RESULTS_DIR / f"{slugify(narrative)}_SKIPPED.txt"
    output_path.write_text(
        f"Skipped narrative: {narrative}\nReason: {reason}\n",
        encoding="utf-8",
    )


def save_topic_visualization(model: BERTopic, narrative: str) -> None:
    try:
        figure = model.visualize_topics()
        output_path = FIGURE_DIR / f"{slugify(narrative)}_topics.html"
        figure.write_html(str(output_path))
        print(f"Saved topic visualization: {output_path}")
    except Exception as error:
        print(f"Warning: could not save topic visualization for {narrative}: {error}")


def run_bertopic_for_narrative(df: pd.DataFrame, narrative: str) -> None:
    narrative_df = df[df["narrative"] == narrative].copy()
    n_docs = len(narrative_df)

    print(f"\nBERTopic framing: {narrative} ({n_docs:,} tweets)")

    if n_docs < MIN_DOCS_TO_MODEL:
        reason = (
            f"Too few tweets for stable BERTopic modeling "
            f"(n={n_docs} < {MIN_DOCS_TO_MODEL})."
        )
        print(f"Skipped: {reason}")
        write_skip_note(narrative, reason)
        return

    texts = narrative_df[TEXT_COLUMN].tolist()

    model = build_topic_model(n_docs)
    topics, _ = model.fit_transform(texts)

    frame_info = model.get_topic_info().copy()
    frame_info["narrative"] = narrative
    frame_info["n_tweets_modeled"] = n_docs

    tweets_with_frames = narrative_df.copy()
    tweets_with_frames["frame"] = topics

    slug = slugify(narrative)

    model_path = MODEL_DIR / f"{slug}.pkl"
    frame_info_path = RESULTS_DIR / f"{slug}_frame_info.csv"
    tweets_path = RESULTS_DIR / f"{slug}_tweets_with_frames.csv"

    model.save(str(model_path))
    frame_info.to_csv(frame_info_path, index=False, encoding="utf-8")
    tweets_with_frames.to_csv(tweets_path, index=False, encoding="utf-8")

    print(f"Saved model:              {model_path}")
    print(f"Saved frame info:         {frame_info_path}")
    print(f"Saved tweets with frames: {tweets_path}")

    save_topic_visualization(model, narrative)


def main() -> None:
    ensure_output_dirs()

    df = load_science_tweets(INPUT_PATH)
    single_label_df = prepare_single_label_tweets(df)

    narratives = sorted(single_label_df["narrative"].unique().tolist())

    print(f"Loaded science subcorpus: {len(df):,} tweets")
    print(f"Single-label tweets:      {len(single_label_df):,}")
    print(f"Narratives to model:      {len(narratives):,}")
    print(f"Narratives:               {narratives}")

    for narrative in narratives:
        try:
            run_bertopic_for_narrative(single_label_df, narrative)
        except Exception as error:
            print(f"Error while modeling {narrative}: {error}")
            write_skip_note(narrative, f"Runtime error: {error}")

    print("\nDone")
    print(f"Models:  {MODEL_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()