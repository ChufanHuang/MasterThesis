#!/usr/bin/env python3
"""
Analyze temporal alignment between sentiment and standing.

Thesis section:
    4.4.2 Temporal Alignment of Sentiment and Standing

Inputs:
    results/10_sentiment_baseline/sentiment_monthly_agg.csv
    results/05_relative_standing/standing_monthly_relative_label_based.csv

Outputs:
    results/11_sentiment_standing_alignment/sentiment_standing_alignment_summary.csv
    figures/11_sentiment_standing_alignment/sentiment_vs_relative_standing_grid.png
    figures/11_sentiment_standing_alignment/sentiment_vs_absolute_standing_grid.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SENTIMENT_MONTHLY_PATH = (
    PROJECT_ROOT
    / "results"
    / "10_sentiment_baseline"
    / "sentiment_monthly_agg.csv"
)

STANDING_MONTHLY_PATH = (
    PROJECT_ROOT
    / "results"
    / "05_relative_standing"
    / "standing_monthly_relative_label_based.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "11_sentiment_standing_alignment"
FIGURE_DIR = PROJECT_ROOT / "figures" / "11_sentiment_standing_alignment"

ALIGNMENT_SUMMARY_PATH = RESULTS_DIR / "sentiment_standing_alignment_summary.csv"
RELATIVE_GRID_PATH = FIGURE_DIR / "sentiment_vs_relative_standing_grid.png"
ABSOLUTE_GRID_PATH = FIGURE_DIR / "sentiment_vs_absolute_standing_grid.png"


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mean = numeric.mean()
    std = numeric.std(ddof=0)

    if std == 0 or np.isnan(std):
        return pd.Series([np.nan] * len(numeric), index=numeric.index)

    return (numeric - mean) / std


def load_sentiment_monthly(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Sentiment monthly file not found: {path}. "
            "Run scripts/10_sentiment_baseline.py first."
        )

    df = pd.read_csv(path, low_memory=False)

    required_columns = {"narrative", "year_month", "avg_valence", "tweet_count"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required sentiment columns: {sorted(missing_columns)}")

    df["month"] = pd.to_datetime(df["year_month"], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["month"]).copy()

    return (
        df[
            [
                "narrative",
                "year_month",
                "month",
                "tweet_count",
                "avg_valence",
            ]
        ]
        .sort_values(["narrative", "month"])
        .reset_index(drop=True)
    )


def load_standing_monthly(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Standing monthly file not found: {path}. "
            "Run scripts/05_relative_standing.py first."
        )

    df = pd.read_csv(path, low_memory=False)

    required_columns = {
        "narrative",
        "year_month",
        "absolute_standing",
        "relative_share",
        "relative_share_percent",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required standing columns: {sorted(missing_columns)}")

    df["month"] = pd.to_datetime(df["year_month"], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["month"]).copy()

    output = df[["narrative", "year_month", "month"]].copy()
    output["standing_share"] = pd.to_numeric(df["relative_share"], errors="coerce")
    output["standing_share_percent"] = pd.to_numeric(
        df["relative_share_percent"],
        errors="coerce",
    )
    output["standing_count"] = pd.to_numeric(df["absolute_standing"], errors="coerce")

    return output.sort_values(["narrative", "month"]).reset_index(drop=True)


def get_narrative_order(sentiment: pd.DataFrame) -> list[str]:
    return (
        sentiment.groupby("narrative")["tweet_count"]
        .sum()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )


def fill_month_index(
    df: pd.DataFrame,
    month_min: pd.Timestamp,
    month_max: pd.Timestamp,
) -> pd.DataFrame:
    full_months = pd.date_range(month_min, month_max, freq="MS")
    base = pd.DataFrame({"month": full_months})
    base["year_month"] = base["month"].dt.strftime("%Y-%m")

    keep_columns = [column for column in df.columns if column not in ["month", "year_month"]]

    merged = base.merge(
        df[["month", "year_month"] + keep_columns],
        on=["month", "year_month"],
        how="left",
    )

    if "narrative" in merged.columns:
        merged["narrative"] = merged["narrative"].ffill().bfill()

    return merged


def compute_alignment_summary(
    merged: pd.DataFrame,
    standing_mode: str,
) -> pd.DataFrame:
    standing_column = "standing_share" if standing_mode == "share" else "standing_count"

    rows = []

    for narrative, group in merged.groupby("narrative"):
        group = group.sort_values("month").copy()
        valid = group.dropna(subset=[standing_column, "avg_valence"]).copy()

        if len(valid) < 6:
            continue

        valid["standing_z"] = zscore(valid[standing_column])
        valid["sentiment_z"] = zscore(valid["avg_valence"])

        corr0 = valid["standing_z"].corr(valid["sentiment_z"])
        corr_lag1 = valid["standing_z"].shift(1).corr(valid["sentiment_z"])
        corr_lead1 = valid["sentiment_z"].shift(1).corr(valid["standing_z"])
        same_direction_share = float(
            np.mean(np.sign(valid["standing_z"]) == np.sign(valid["sentiment_z"]))
        )

        rows.append(
            {
                "narrative": narrative,
                "n_months_used": len(valid),
                "standing_mode": standing_mode,
                "corr0_z": round(float(corr0), 3) if pd.notna(corr0) else np.nan,
                "corr_lag1_z": round(float(corr_lag1), 3)
                if pd.notna(corr_lag1)
                else np.nan,
                "corr_lead1_z": round(float(corr_lead1), 3)
                if pd.notna(corr_lead1)
                else np.nan,
                "same_direction_share": round(float(same_direction_share), 3),
            }
        )

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary = summary.sort_values(
            ["standing_mode", "corr0_z"],
            ascending=[True, False],
        ).reset_index(drop=True)

    return summary


def plot_overlay_grid(
    merged: pd.DataFrame,
    order: list[str],
    standing_mode: str,
    output_path: Path,
    n_columns: int = 2,
) -> None:
    standing_column = "standing_share" if standing_mode == "share" else "standing_count"

    month_min = merged["month"].min()
    month_max = merged["month"].max()

    n_items = len(order)
    n_rows = int(np.ceil(n_items / n_columns))

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_columns,
        figsize=(16, 3.2 * n_rows),
        sharex=True,
        sharey=True,
    )

    axes = np.array(axes).reshape(-1)

    for index, narrative in enumerate(order):
        ax = axes[index]
        group = merged[merged["narrative"] == narrative].copy()

        if group.empty:
            ax.set_axis_off()
            continue

        group = fill_month_index(group, month_min, month_max)
        group["standing_z"] = zscore(group[standing_column])
        group["sentiment_z"] = zscore(group["avg_valence"])

        ax.plot(
            group["month"],
            group["standing_z"],
            linewidth=1.6,
            label="Standing (z)",
        )
        ax.plot(
            group["month"],
            group["sentiment_z"],
            linewidth=1.6,
            label="Sentiment (z)",
        )
        ax.axhline(0, linewidth=1, color="black", alpha=0.6)

        valid = group.dropna(subset=["standing_z", "sentiment_z"])
        corr0 = valid["standing_z"].corr(valid["sentiment_z"]) if len(valid) >= 6 else np.nan
        corr_text = f"r={corr0:.2f}" if pd.notna(corr0) else "r=NA"

        ax.set_title(f"{narrative} ({corr_text})", fontsize=11)
        ax.set_ylabel("Z-score")
        ax.set_xlabel("Month")
        ax.grid(True, linestyle="--", alpha=0.35)

    for index in range(n_items, len(axes)):
        axes[index].set_axis_off()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)

    fig.suptitle(
        "Sentiment-Standing Alignment Over Time",
        fontsize=14,
        y=0.995,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()

    sentiment = load_sentiment_monthly(SENTIMENT_MONTHLY_PATH)
    standing = load_standing_monthly(STANDING_MONTHLY_PATH)

    order = get_narrative_order(sentiment)

    merged = sentiment.merge(
        standing,
        on=["narrative", "year_month", "month"],
        how="left",
    )

    summary_parts = []

    if merged["standing_share"].notna().any():
        plot_overlay_grid(merged, order, "share", RELATIVE_GRID_PATH, n_columns=2)
        summary_parts.append(compute_alignment_summary(merged, "share"))

    if merged["standing_count"].notna().any():
        plot_overlay_grid(merged, order, "count", ABSOLUTE_GRID_PATH, n_columns=2)
        summary_parts.append(compute_alignment_summary(merged, "count"))

    if summary_parts:
        summary = pd.concat(summary_parts, ignore_index=True)
        summary.to_csv(ALIGNMENT_SUMMARY_PATH, index=False, encoding="utf-8")
    else:
        summary = pd.DataFrame()

    print("\nSaved outputs")
    print(f"Alignment summary: {ALIGNMENT_SUMMARY_PATH}")
    print(f"Relative grid:     {RELATIVE_GRID_PATH}")
    print(f"Absolute grid:     {ABSOLUTE_GRID_PATH}")

    if not summary.empty:
        print("\nSummary preview")
        print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()