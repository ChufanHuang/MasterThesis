#!/usr/bin/env python3
"""
Compute framing architecture types from narrative-specific BERTopic outputs.

Thesis section:
    4.3.2 Frame Architecture Types

Inputs:
    results/07_bertopic_framing/*_frame_info.csv

Outputs:
    results/08_frame_architecture_types/frame_architecture_typology.csv
    results/08_frame_architecture_types/frame_architecture_type_counts.csv
    figures/08_frame_architecture_types/frame_architecture_map.png

Notes:
    Structure statistics are computed after removing BERTopic outliers
    where Topic = -1.

Typology rules:
    1. Highly fragmented:     effective_frames >= 14
    2. Ultra-centralized:     top1_share_percent >= 60
    3. Centralized:           top1_share_percent >= 30
    4. Quasi-centralized:     top1_share_percent >= 20
    5. Moderately structured: otherwise
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "results" / "07_bertopic_framing"

RESULTS_DIR = PROJECT_ROOT / "results" / "08_frame_architecture_types"
FIGURE_DIR = PROJECT_ROOT / "figures" / "08_frame_architecture_types"

TYPOLOGY_TABLE_PATH = RESULTS_DIR / "frame_architecture_typology.csv"
TYPE_COUNTS_PATH = RESULTS_DIR / "frame_architecture_type_counts.csv"
ARCHITECTURE_MAP_PATH = FIGURE_DIR / "frame_architecture_map.png"

TYPE_ORDER = [
    "Ultra-centralized",
    "Centralized",
    "Quasi-centralized",
    "Moderately structured",
    "Highly fragmented",
    "No inlier frames",
]

TYPE_RANK = {architecture_type: index for index, architecture_type in enumerate(TYPE_ORDER)}


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def assign_architecture_type(
    top1_share_percent: float,
    effective_frames: float,
) -> str:
    if pd.notna(effective_frames) and effective_frames >= 14:
        return "Highly fragmented"

    if pd.notna(top1_share_percent) and top1_share_percent >= 60:
        return "Ultra-centralized"

    if pd.notna(top1_share_percent) and top1_share_percent >= 30:
        return "Centralized"

    if pd.notna(top1_share_percent) and top1_share_percent >= 20:
        return "Quasi-centralized"

    return "Moderately structured"


def get_narrative_name(frame_info: pd.DataFrame, path: Path) -> str:
    if "narrative" in frame_info.columns and frame_info["narrative"].notna().any():
        return str(frame_info.loc[frame_info["narrative"].notna(), "narrative"].iloc[0]).strip()

    return path.stem.replace("_frame_info", "").strip()


def load_frame_info(path: Path) -> pd.DataFrame:
    frame_info = pd.read_csv(path)

    required_columns = {"Topic", "Count"}
    missing_columns = required_columns - set(frame_info.columns)
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing_columns)}")

    frame_info = frame_info.copy()
    frame_info["Topic"] = pd.to_numeric(frame_info["Topic"], errors="coerce")
    frame_info["Count"] = (
        pd.to_numeric(frame_info["Count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    frame_info = frame_info.dropna(subset=["Topic"]).copy()
    frame_info["Topic"] = frame_info["Topic"].astype(int)

    return frame_info


def compute_architecture_for_file(path: Path) -> dict[str, object] | None:
    frame_info = load_frame_info(path)
    narrative = get_narrative_name(frame_info, path)

    n_total = int(frame_info["Count"].sum())
    if n_total <= 0:
        return None

    outlier_count = int(frame_info.loc[frame_info["Topic"] == -1, "Count"].sum())
    outlier_share_percent = outlier_count / n_total * 100

    inlier_frames = frame_info[frame_info["Topic"] != -1].copy()
    n_inlier = int(inlier_frames["Count"].sum())

    if n_inlier <= 0:
        return {
            "narrative": narrative,
            "n_total": n_total,
            "n_inlier": 0,
            "outlier_count": outlier_count,
            "outlier_share_percent": round(outlier_share_percent, 2),
            "n_frames": 0,
            "top1_share_percent": np.nan,
            "top3_share_percent": np.nan,
            "hhi": np.nan,
            "effective_frames": np.nan,
            "architecture_type": "No inlier frames",
        }

    inlier_frames["frame_share"] = inlier_frames["Count"] / n_inlier
    frame_shares = inlier_frames["frame_share"].sort_values(ascending=False).to_numpy()

    n_frames = int(inlier_frames["Topic"].nunique())
    top1_share_percent = float(frame_shares[0] * 100)
    top3_share_percent = float(frame_shares[:3].sum() * 100)
    hhi = float(np.sum(frame_shares**2))
    effective_frames = float(1.0 / hhi) if hhi > 0 else np.nan

    architecture_type = assign_architecture_type(
        top1_share_percent=top1_share_percent,
        effective_frames=effective_frames,
    )

    return {
        "narrative": narrative,
        "n_total": n_total,
        "n_inlier": n_inlier,
        "outlier_count": outlier_count,
        "outlier_share_percent": round(outlier_share_percent, 2),
        "n_frames": n_frames,
        "top1_share_percent": round(top1_share_percent, 2),
        "top3_share_percent": round(top3_share_percent, 2),
        "hhi": round(hhi, 4),
        "effective_frames": round(effective_frames, 2),
        "architecture_type": architecture_type,
    }


def build_typology_table(input_dir: Path) -> pd.DataFrame:
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}. "
            "Run scripts/07_bertopic_framing.py first."
        )

    frame_info_files = sorted(input_dir.glob("*_frame_info.csv"))
    if not frame_info_files:
        raise FileNotFoundError(f"No *_frame_info.csv files found in: {input_dir}")

    rows = []
    for path in frame_info_files:
        row = compute_architecture_for_file(path)
        if row is not None:
            rows.append(row)

    if not rows:
        raise ValueError("No valid frame architecture rows could be computed.")

    table = pd.DataFrame(rows)
    table["_type_rank"] = table["architecture_type"].map(TYPE_RANK).fillna(99).astype(int)

    table = (
        table.sort_values(
            by=[
                "_type_rank",
                "top1_share_percent",
                "n_frames",
                "n_total",
            ],
            ascending=[True, False, False, False],
        )
        .drop(columns="_type_rank")
        .reset_index(drop=True)
    )

    return table


def build_type_counts(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.groupby("architecture_type", as_index=False)
        .agg(
            n_narratives=("narrative", "nunique"),
            total_docs=("n_total", "sum"),
            mean_top1_share_percent=("top1_share_percent", "mean"),
            mean_effective_frames=("effective_frames", "mean"),
        )
        .round(
            {
                "mean_top1_share_percent": 2,
                "mean_effective_frames": 2,
            }
        )
        .sort_values("n_narratives", ascending=False)
        .reset_index(drop=True)
    )


def plot_architecture_map(table: pd.DataFrame) -> None:
    plot_df = table.copy()
    plot_df = plot_df[plot_df["n_frames"] > 0].copy()
    plot_df = plot_df.dropna(subset=["top1_share_percent", "effective_frames"])

    fig, ax = plt.subplots(figsize=(12, 7))

    for architecture_type in TYPE_ORDER:
        subset = plot_df[plot_df["architecture_type"] == architecture_type]
        if subset.empty:
            continue

        ax.scatter(
            subset["top1_share_percent"],
            subset["effective_frames"],
            label=architecture_type,
        )

        for _, row in subset.iterrows():
            ax.text(
                float(row["top1_share_percent"]) + 0.6,
                float(row["effective_frames"]) + 0.08,
                str(row["narrative"]),
                fontsize=9,
            )

    ax.axvline(60, linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(30, linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(20, linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(14, linestyle="--", linewidth=1, alpha=0.6)

    ax.set_title("Framing Architecture Map", fontweight="bold")
    ax.set_xlabel("Top-1 frame share (%)")
    ax.set_ylabel("Effective number of frames")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    fig.savefig(ARCHITECTURE_MAP_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()

    typology_table = build_typology_table(INPUT_DIR)
    type_counts = build_type_counts(typology_table)

    typology_table.to_csv(TYPOLOGY_TABLE_PATH, index=False, encoding="utf-8")
    type_counts.to_csv(TYPE_COUNTS_PATH, index=False, encoding="utf-8")
    plot_architecture_map(typology_table)

    print("\nSaved outputs")
    print(f"Typology table: {TYPOLOGY_TABLE_PATH}")
    print(f"Type counts:    {TYPE_COUNTS_PATH}")
    print(f"Figure:         {ARCHITECTURE_MAP_PATH}")

    print("\nPreview")
    print(
        typology_table[
            [
                "narrative",
                "n_total",
                "n_frames",
                "top1_share_percent",
                "top3_share_percent",
                "effective_frames",
                "architecture_type",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()