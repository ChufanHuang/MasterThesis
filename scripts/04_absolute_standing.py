#!/usr/bin/env python3
"""
Compute and visualize monthly absolute standing.

Thesis section:
    4.2.1 Monthly Absolute Standing of Science-Related Conspiracy Narratives

Inputs:
    data/processed/science_conspiracy_tweets.csv

Outputs:
    data/processed/standing_monthly_by_narrative_label.csv
    data/processed/standing_monthly_total_unique_tweets.csv
    data/processed/standing_monthly_total_label_counts.csv
    results/04_absolute_standing/absolute_standing_trend_summary.csv
    figures/04_absolute_standing/total_unique_tweets.png
    figures/04_absolute_standing/flat_earth_vs_others.png
    figures/04_absolute_standing/selected_narratives.png
"""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "science_conspiracy_tweets.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "04_absolute_standing"
FIGURE_DIR = PROJECT_ROOT / "figures" / "04_absolute_standing"

MONTHLY_BY_NARRATIVE_PATH = PROCESSED_DIR / "standing_monthly_by_narrative_label.csv"
MONTHLY_TOTAL_UNIQUE_PATH = PROCESSED_DIR / "standing_monthly_total_unique_tweets.csv"
MONTHLY_TOTAL_LABEL_PATH = PROCESSED_DIR / "standing_monthly_total_label_counts.csv"
TREND_SUMMARY_PATH = RESULTS_DIR / "absolute_standing_trend_summary.csv"

TOTAL_UNIQUE_FIGURE_PATH = FIGURE_DIR / "total_unique_tweets.png"
DOMINANT_VS_OTHERS_FIGURE_PATH = FIGURE_DIR / "flat_earth_vs_others.png"
SELECTED_NARRATIVES_FIGURE_PATH = FIGURE_DIR / "selected_narratives.png"

DOMINANT_NARRATIVE = "Flat Earth"
FOCUS_NON_DOMINANT_NARRATIVES = [
    "Fluoride_Cancer_Misc",
    "Geoengineering",
    "5G",
    "Chemtrails",
]
HIGHLIGHT_WINDOW = ("2018-08", "2018-12")
FIGURE_SIZE_WIDE = (12, 5)
FIGURE_SIZE_TALL = (12, 6)


def ensure_output_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_science_tweets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. "
            "Run scripts/03_dictionary_label_full.py first."
        )

    df = pd.read_csv(path, low_memory=False)

    required_columns = {"created_at", "science_narratives"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return df


def parse_narratives(value: object) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            pass

    return [text]


def prepare_monthly_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    parsed_dates = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.loc[parsed_dates.notna()].copy()
    df["created_at"] = parsed_dates.loc[parsed_dates.notna()]
    df["year_month"] = df["created_at"].dt.to_period("M").astype(str)

    if "status_id" in df.columns and df["status_id"].notna().any():
        df["tweet_identifier"] = df["status_id"].astype(str)
    elif "tweet_id" in df.columns and df["tweet_id"].notna().any():
        df["tweet_identifier"] = df["tweet_id"].astype(str)
    else:
        df = df.reset_index(drop=True)
        df["tweet_identifier"] = df.index.astype(str)

    df["narratives_list"] = df["science_narratives"].apply(parse_narratives)

    return df


def compute_monthly_total_unique(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("year_month")["tweet_identifier"]
        .nunique()
        .reset_index(name="total_science_tweets")
        .sort_values("year_month")
        .reset_index(drop=True)
    )


def compute_monthly_by_narrative(df: pd.DataFrame) -> pd.DataFrame:
    exploded = df.explode("narratives_list").rename(
        columns={"narratives_list": "narrative"}
    )
    exploded["narrative"] = exploded["narrative"].fillna("").astype(str).str.strip()
    exploded = exploded[exploded["narrative"].str.len() > 0].copy()

    return (
        exploded.groupby(["year_month", "narrative"], as_index=False)
        .size()
        .rename(columns={"size": "absolute_standing"})
        .sort_values(["year_month", "absolute_standing"], ascending=[True, False])
        .reset_index(drop=True)
    )


def compute_monthly_total_labels(monthly_by_narrative: pd.DataFrame) -> pd.DataFrame:
    return (
        monthly_by_narrative.groupby("year_month")["absolute_standing"]
        .sum()
        .reset_index(name="total_label_counts")
        .sort_values("year_month")
        .reset_index(drop=True)
    )


def build_trend_summary(monthly_by_narrative: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for narrative, group in monthly_by_narrative.groupby("narrative"):
        group = group.sort_values("year_month").reset_index(drop=True)
        peak_row = group.loc[group["absolute_standing"].idxmax()]

        first_value = int(group.iloc[0]["absolute_standing"])
        last_value = int(group.iloc[-1]["absolute_standing"])

        if last_value > first_value:
            trend_direction = "increase"
        elif last_value < first_value:
            trend_direction = "decrease"
        else:
            trend_direction = "stable"

        rows.append(
            {
                "narrative": narrative,
                "first_active_month": group.iloc[0]["year_month"],
                "last_active_month": group.iloc[-1]["year_month"],
                "peak_month": peak_row["year_month"],
                "peak_value": int(peak_row["absolute_standing"]),
                "total_labels": int(group["absolute_standing"].sum()),
                "active_months": int(group["year_month"].nunique()),
                "mean_monthly_standing": round(
                    float(group["absolute_standing"].mean()), 2
                ),
                "first_active_value": first_value,
                "last_active_value": last_value,
                "trend_direction_first_to_last": trend_direction,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("total_labels", ascending=False)
        .reset_index(drop=True)
    )


def month_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y-%m", errors="coerce")


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_highlight_window(ax: plt.Axes) -> None:
    if HIGHLIGHT_WINDOW is None:
        return

    start, end = HIGHLIGHT_WINDOW
    ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), alpha=0.12)


def save_total_unique_figure(monthly_total_unique: pd.DataFrame) -> None:
    df = monthly_total_unique.copy()
    df["year_month_dt"] = month_to_datetime(df["year_month"])
    df = df.dropna(subset=["year_month_dt"]).sort_values("year_month_dt")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    ax.plot(
        df["year_month_dt"],
        df["total_science_tweets"],
        linewidth=2.2,
        marker="o",
        markersize=3,
        label="Total science-related tweets",
    )

    if not df.empty:
        peak = df.loc[df["total_science_tweets"].idxmax()]
        ax.annotate(
            f"Peak ({peak['year_month_dt'].strftime('%Y-%m')}): "
            f"{int(peak['total_science_tweets'])}",
            xy=(peak["year_month_dt"], peak["total_science_tweets"]),
            xytext=(peak["year_month_dt"], peak["total_science_tweets"] * 0.7),
            arrowprops={"arrowstyle": "->", "linewidth": 1},
            fontsize=9,
            ha="left",
        )

    ax.set_title(
        "Monthly Standing of Science-Related Conspiracy Tweets",
        fontweight="bold",
    )
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Number of Tweets")
    style_axis(ax)
    add_highlight_window(ax)
    ax.legend(loc="upper left", frameon=False)

    plt.tight_layout()
    fig.savefig(TOTAL_UNIQUE_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_dominant_vs_others_figure(monthly_by_narrative: pd.DataFrame) -> None:
    df = monthly_by_narrative.copy()
    df["year_month_dt"] = month_to_datetime(df["year_month"])
    df = df.dropna(subset=["year_month_dt"])

    dominant = (
        df[df["narrative"] == DOMINANT_NARRATIVE]
        .groupby("year_month_dt", as_index=False)["absolute_standing"]
        .sum()
        .rename(columns={"absolute_standing": "dominant_labels"})
    )

    others = (
        df[df["narrative"] != DOMINANT_NARRATIVE]
        .groupby("year_month_dt", as_index=False)["absolute_standing"]
        .sum()
        .rename(columns={"absolute_standing": "other_labels"})
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    ax.plot(
        dominant["year_month_dt"],
        dominant["dominant_labels"],
        label=f"{DOMINANT_NARRATIVE}",
        linewidth=2.4,
        marker="o",
        markersize=3,
    )
    ax.plot(
        others["year_month_dt"],
        others["other_labels"],
        label="All other narratives",
        linewidth=2,
        linestyle="--",
        marker="s",
        markersize=3,
    )

    if not dominant.empty:
        peak = dominant.loc[dominant["dominant_labels"].idxmax()]
        ax.annotate(
            f"Peak ({peak['year_month_dt'].strftime('%Y-%m')}): "
            f"{int(peak['dominant_labels'])}",
            xy=(peak["year_month_dt"], peak["dominant_labels"]),
            xytext=(peak["year_month_dt"], peak["dominant_labels"] * 0.7),
            arrowprops={"arrowstyle": "->", "linewidth": 1},
            fontsize=9,
            ha="left",
        )

    ax.set_title(
        "Monthly Standing: Flat Earth vs. All Other Narratives",
        fontweight="bold",
    )
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Number of Labels")
    style_axis(ax)
    add_highlight_window(ax)
    ax.legend(loc="upper left", frameon=False)

    plt.tight_layout()
    fig.savefig(DOMINANT_VS_OTHERS_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_selected_narratives(monthly_by_narrative: pd.DataFrame) -> list[str]:
    available = set(monthly_by_narrative["narrative"])

    selected = [
        narrative
        for narrative in FOCUS_NON_DOMINANT_NARRATIVES
        if narrative in available and narrative != DOMINANT_NARRATIVE
    ]

    if selected:
        return selected

    totals = (
        monthly_by_narrative[
            monthly_by_narrative["narrative"] != DOMINANT_NARRATIVE
        ]
        .groupby("narrative")["absolute_standing"]
        .sum()
        .sort_values(ascending=False)
    )

    return totals.head(4).index.tolist()


def save_selected_narratives_figure(monthly_by_narrative: pd.DataFrame) -> None:
    df = monthly_by_narrative.copy()
    df["year_month_dt"] = month_to_datetime(df["year_month"])
    df = df.dropna(subset=["year_month_dt"])

    selected_narratives = get_selected_narratives(df)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_TALL)

    for narrative in selected_narratives:
        subset = (
            df[df["narrative"] == narrative]
            .groupby("year_month_dt", as_index=False)["absolute_standing"]
            .sum()
            .sort_values("year_month_dt")
        )

        ax.plot(
            subset["year_month_dt"],
            subset["absolute_standing"],
            label=narrative,
            linewidth=2,
            marker="o",
            markersize=3,
        )

    ax.set_title("Monthly Standing of Selected Narratives", fontweight="bold")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Number of Labels")
    style_axis(ax)
    add_highlight_window(ax)
    ax.legend(loc="upper left", frameon=False)

    plt.tight_layout()
    fig.savefig(SELECTED_NARRATIVES_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Selected narratives: {selected_narratives}")


def main() -> None:
    ensure_output_dirs()

    df = load_science_tweets(INPUT_PATH)
    df = prepare_monthly_data(df)

    monthly_total_unique = compute_monthly_total_unique(df)
    monthly_by_narrative = compute_monthly_by_narrative(df)
    monthly_total_labels = compute_monthly_total_labels(monthly_by_narrative)
    trend_summary = build_trend_summary(monthly_by_narrative)

    monthly_total_unique.to_csv(
        MONTHLY_TOTAL_UNIQUE_PATH,
        index=False,
        encoding="utf-8",
    )
    monthly_by_narrative.to_csv(
        MONTHLY_BY_NARRATIVE_PATH,
        index=False,
        encoding="utf-8",
    )
    monthly_total_labels.to_csv(
        MONTHLY_TOTAL_LABEL_PATH,
        index=False,
        encoding="utf-8",
    )
    trend_summary.to_csv(TREND_SUMMARY_PATH, index=False, encoding="utf-8")

    save_total_unique_figure(monthly_total_unique)
    save_dominant_vs_others_figure(monthly_by_narrative)
    save_selected_narratives_figure(monthly_by_narrative)

    print("\nSaved outputs")
    print(f"Monthly by narrative: {MONTHLY_BY_NARRATIVE_PATH}")
    print(f"Monthly unique total: {MONTHLY_TOTAL_UNIQUE_PATH}")
    print(f"Monthly label total:  {MONTHLY_TOTAL_LABEL_PATH}")
    print(f"Trend summary:        {TREND_SUMMARY_PATH}")
    print(f"Figure:               {TOTAL_UNIQUE_FIGURE_PATH}")
    print(f"Figure:               {DOMINANT_VS_OTHERS_FIGURE_PATH}")
    print(f"Figure:               {SELECTED_NARRATIVES_FIGURE_PATH}")


if __name__ == "__main__":
    main()