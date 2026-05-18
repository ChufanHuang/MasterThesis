#!/usr/bin/env python3
"""
Compute and visualize monthly dominance and competition dynamics.

Thesis section:
    4.2.3 Dominance and Competitive Dynamics

Inputs:
    results/05_relative_standing/standing_monthly_relative_label_based.csv

Outputs:
    results/06_dominance_competition/dominance_monthly_label_based.csv
    figures/06_dominance_competition/dominance_indices.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "05_relative_standing"
    / "standing_monthly_relative_label_based.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "06_dominance_competition"
FIGURE_DIR = PROJECT_ROOT / "figures" / "06_dominance_competition"

DOMINANCE_TABLE_PATH = RESULTS_DIR / "dominance_monthly_label_based.csv"
DOMINANCE_FIGURE_PATH = FIGURE_DIR / "dominance_indices.png"

FIGURE_SIZE = (12, 9)


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_relative_standing(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. "
            "Run scripts/05_relative_standing.py first."
        )

    df = pd.read_csv(path)

    required_columns = {"year_month", "narrative", "relative_share"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["year_month_dt"] = pd.to_datetime(
        df["year_month"],
        format="%Y-%m",
        errors="coerce",
    )
    df = df.dropna(subset=["year_month_dt"]).copy()

    df["narrative"] = df["narrative"].astype(str).str.strip()
    df["relative_share"] = pd.to_numeric(
        df["relative_share"],
        errors="coerce",
    ).fillna(0.0)

    return df


def compute_dominance_indices(relative: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for month, group in relative.groupby("year_month_dt"):
        shares = group["relative_share"].to_numpy(dtype=float)
        share_sum = shares.sum()

        if share_sum <= 0:
            continue

        shares = shares / share_sum
        shares_sorted = np.sort(shares)[::-1]

        hhi = float(np.sum(shares**2))
        effective_narratives = float(1.0 / hhi) if hhi > 0 else np.nan

        rows.append(
            {
                "year_month": month.strftime("%Y-%m"),
                "top1_share": float(shares_sorted[0]),
                "top1_share_percent": float(shares_sorted[0] * 100),
                "top3_share": float(shares_sorted[:3].sum())
                if len(shares_sorted) >= 3
                else float(shares_sorted.sum()),
                "top3_share_percent": float(shares_sorted[:3].sum() * 100)
                if len(shares_sorted) >= 3
                else float(shares_sorted.sum() * 100),
                "hhi": hhi,
                "effective_narratives": effective_narratives,
                "n_active_narratives": int((shares > 0).sum()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("year_month")
        .reset_index(drop=True)
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_dominance_figure(dominance: pd.DataFrame) -> None:
    df = dominance.copy()
    df["year_month_dt"] = pd.to_datetime(
        df["year_month"],
        format="%Y-%m",
        errors="coerce",
    )
    df = df.dropna(subset=["year_month_dt"])

    fig, axes = plt.subplots(3, 1, figsize=FIGURE_SIZE, sharex=True)

    axes[0].plot(
        df["year_month_dt"],
        df["top1_share_percent"],
        linewidth=2.4,
        marker="o",
        markersize=3,
        label="Top-1 share",
    )
    axes[0].plot(
        df["year_month_dt"],
        df["top3_share_percent"],
        linewidth=2,
        linestyle="--",
        marker="s",
        markersize=3,
        label="Top-3 share",
    )
    axes[0].set_ylabel("Share (%)")
    axes[0].set_title("Top Narrative Concentration", fontweight="bold")
    axes[0].legend(loc="upper left", frameon=False)
    style_axis(axes[0])

    axes[1].plot(
        df["year_month_dt"],
        df["hhi"],
        linewidth=2.2,
        marker="o",
        markersize=3,
        color="black",
        label="HHI",
    )
    axes[1].set_ylabel("HHI")
    axes[1].set_title("Concentration Index", fontweight="bold")
    axes[1].legend(loc="upper left", frameon=False)
    style_axis(axes[1])

    axes[2].plot(
        df["year_month_dt"],
        df["effective_narratives"],
        linewidth=2.2,
        marker="o",
        markersize=3,
        label="Effective number of narratives",
    )
    axes[2].set_ylabel("1 / HHI")
    axes[2].set_xlabel("Year-Month")
    axes[2].set_title("Effective Competition", fontweight="bold")
    axes[2].legend(loc="upper left", frameon=False)
    style_axis(axes[2])

    fig.suptitle(
        "Dominance and Competition Dynamics over Time",
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    fig.savefig(DOMINANCE_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()

    relative = load_relative_standing(INPUT_PATH)
    dominance = compute_dominance_indices(relative)

    dominance.to_csv(DOMINANCE_TABLE_PATH, index=False, encoding="utf-8")
    save_dominance_figure(dominance)

    print("\nSaved outputs")
    print(f"Dominance table: {DOMINANCE_TABLE_PATH}")
    print(f"Figure:          {DOMINANCE_FIGURE_PATH}")


if __name__ == "__main__":
    main()