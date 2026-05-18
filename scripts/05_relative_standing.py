#!/usr/bin/env python3
"""
Compute and visualize monthly relative standing.

Thesis section:
    4.2.2 Monthly Relative Standing of Science-Related Conspiracy Narratives

Inputs:
    data/processed/standing_monthly_by_narrative_label.csv

Outputs:
    results/05_relative_standing/standing_monthly_relative_label_based.csv
    figures/05_relative_standing/flat_earth_relative_share.png
    figures/05_relative_standing/relative_share_area_top_n.png
    figures/05_relative_standing/relative_share_area_all.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "standing_monthly_by_narrative_label.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "05_relative_standing"
FIGURE_DIR = PROJECT_ROOT / "figures" / "05_relative_standing"

RELATIVE_STANDING_PATH = RESULTS_DIR / "standing_monthly_relative_label_based.csv"
DOMINANT_SHARE_FIGURE_PATH = FIGURE_DIR / "flat_earth_relative_share.png"
TOP_N_AREA_FIGURE_PATH = FIGURE_DIR / "relative_share_area_top_n.png"
ALL_AREA_FIGURE_PATH = FIGURE_DIR / "relative_share_area_all.png"

DOMINANT_NARRATIVE = "Flat Earth"
TOP_N_AREA = 6
FIGURE_SIZE = (12, 6)


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def month_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y-%m", errors="coerce")


def load_absolute_standing(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. "
            "Run scripts/04_absolute_standing.py first."
        )

    df = pd.read_csv(path)

    required_columns = {"year_month", "narrative", "absolute_standing"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["year_month_dt"] = month_to_datetime(df["year_month"])
    df = df.dropna(subset=["year_month_dt"]).copy()

    df["narrative"] = df["narrative"].astype(str).str.strip()
    df["absolute_standing"] = (
        pd.to_numeric(df["absolute_standing"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    return df


def compute_relative_standing(df: pd.DataFrame) -> pd.DataFrame:
    monthly_total = (
        df.groupby("year_month_dt", as_index=False)["absolute_standing"]
        .sum()
        .rename(columns={"absolute_standing": "monthly_total_labels"})
    )

    relative = df.merge(monthly_total, on="year_month_dt", how="left")
    denominator = relative["monthly_total_labels"].replace(0, pd.NA)

    relative["relative_share"] = (
        relative["absolute_standing"] / denominator
    ).fillna(0.0)
    relative["relative_share_percent"] = relative["relative_share"] * 100

    relative["year_month"] = relative["year_month_dt"].dt.strftime("%Y-%m")

    return relative[
        [
            "year_month",
            "year_month_dt",
            "narrative",
            "absolute_standing",
            "monthly_total_labels",
            "relative_share",
            "relative_share_percent",
        ]
    ].sort_values(["year_month", "relative_share"], ascending=[True, False])


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_dominant_share_figure(relative: pd.DataFrame) -> None:
    months = (
        relative[["year_month_dt", "monthly_total_labels"]]
        .drop_duplicates()
        .sort_values("year_month_dt")
    )

    dominant = relative.loc[
        relative["narrative"] == DOMINANT_NARRATIVE,
        ["year_month_dt", "relative_share_percent"],
    ]

    dominant = months.merge(dominant, on="year_month_dt", how="left")
    dominant["relative_share_percent"] = dominant[
        "relative_share_percent"
    ].fillna(0.0)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(
        dominant["year_month_dt"],
        dominant["relative_share_percent"],
        linewidth=2.5,
        marker="o",
        markersize=3,
        label=f"{DOMINANT_NARRATIVE} share",
    )

    if not dominant.empty and dominant["relative_share_percent"].max() > 0:
        peak_index = dominant["relative_share_percent"].idxmax()
        peak_x = dominant.loc[peak_index, "year_month_dt"]
        peak_y = dominant.loc[peak_index, "relative_share_percent"]

        ax.annotate(
            f"Peak ({peak_x.strftime('%Y-%m')}): {peak_y:.1f}%",
            xy=(peak_x, peak_y),
            xytext=(peak_x, min(95, peak_y + 15)),
            arrowprops={"arrowstyle": "->", "linewidth": 1},
            fontsize=9,
            ha="left",
        )

    ax.set_title(
        f"Monthly Relative Standing: {DOMINANT_NARRATIVE} Share",
        fontweight="bold",
    )
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Share within science-related labels (%)")
    ax.set_ylim(0, 100)
    style_axis(ax)
    ax.legend(loc="upper left", frameon=False)

    plt.tight_layout()
    fig.savefig(DOMINANT_SHARE_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_area_plot_table(relative: pd.DataFrame, top_n: int | None) -> pd.DataFrame:
    df = relative.copy()

    if top_n is not None:
        totals = (
            df.groupby("narrative", as_index=False)["absolute_standing"]
            .sum()
            .sort_values("absolute_standing", ascending=False)
        )
        top_narratives = totals["narrative"].head(top_n).tolist()
        df["plot_group"] = df["narrative"].where(
            df["narrative"].isin(top_narratives),
            other="Other",
        )
    else:
        top_narratives = (
            df.groupby("narrative")["absolute_standing"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        df["plot_group"] = df["narrative"]

    grouped = (
        df.groupby(["year_month_dt", "plot_group"], as_index=False)[
            "absolute_standing"
        ]
        .sum()
        .merge(
            df.groupby("year_month_dt", as_index=False)[
                "monthly_total_labels"
            ].first(),
            on="year_month_dt",
            how="left",
        )
    )

    denominator = grouped["monthly_total_labels"].replace(0, pd.NA)
    grouped["share_percent"] = (
        grouped["absolute_standing"] / denominator * 100
    ).fillna(0.0)

    wide = grouped.pivot(
        index="year_month_dt",
        columns="plot_group",
        values="share_percent",
    ).fillna(0.0)

    ordered_columns = [column for column in top_narratives if column in wide.columns]

    if DOMINANT_NARRATIVE in ordered_columns:
        ordered_columns = [DOMINANT_NARRATIVE] + [
            column for column in ordered_columns if column != DOMINANT_NARRATIVE
        ]

    if "Other" in wide.columns:
        ordered_columns.append("Other")

    return wide[ordered_columns]


def save_area_figure(
    wide: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.stackplot(
        wide.index,
        wide.T.values,
        labels=wide.columns,
        alpha=0.85,
    )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Share within science-related labels (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()

    absolute = load_absolute_standing(INPUT_PATH)
    relative = compute_relative_standing(absolute)

    relative.to_csv(RELATIVE_STANDING_PATH, index=False, encoding="utf-8")

    save_dominant_share_figure(relative)

    top_n_table = build_area_plot_table(relative, top_n=TOP_N_AREA)
    save_area_figure(
        top_n_table,
        TOP_N_AREA_FIGURE_PATH,
        f"Monthly Relative Standing Structure (Top {TOP_N_AREA} + Other)",
    )

    all_table = build_area_plot_table(relative, top_n=None)
    save_area_figure(
        all_table,
        ALL_AREA_FIGURE_PATH,
        "Monthly Relative Standing Structure (All Narratives)",
    )

    print("\nSaved outputs")
    print(f"Relative standing table: {RELATIVE_STANDING_PATH}")
    print(f"Figure:                  {DOMINANT_SHARE_FIGURE_PATH}")
    print(f"Figure:                  {TOP_N_AREA_FIGURE_PATH}")
    print(f"Figure:                  {ALL_AREA_FIGURE_PATH}")


if __name__ == "__main__":
    main()