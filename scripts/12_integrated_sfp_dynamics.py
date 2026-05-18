#!/usr/bin/env python3
"""
Analyze integrated Standing-Framing-Positioning dynamics.

Thesis section:
    4.4.3 Integrated SFP Dynamics

Inputs:
    results/05_relative_standing/standing_monthly_relative_label_based.csv
    results/10_sentiment_baseline/sentiment_monthly_agg.csv
    results/09_frame_evolution/frames_over_time_all.csv
    results/09_frame_evolution/top_frame_over_time_all.csv
    results/09_frame_evolution/frame_turnover_rate_by_narrative.csv
    results/08_frame_architecture_types/frame_architecture_typology.csv

Outputs:
    results/12_integrated_sfp_dynamics/sfp_master_monthly.csv
    results/12_integrated_sfp_dynamics/dominant_frame_transition_events.csv
    results/12_integrated_sfp_dynamics/sfp_integrated_metrics.csv
    results/12_integrated_sfp_dynamics/sfp_correlation_matrix.csv
    figures/12_integrated_sfp_dynamics/sentiment_frame_coupling_grid.png
    figures/12_integrated_sfp_dynamics/event_centered_sentiment_change.png
    figures/12_integrated_sfp_dynamics/integrated_sfp_timeline.png
    figures/12_integrated_sfp_dynamics/sfp_phase_space.png
    figures/12_integrated_sfp_dynamics/standing_volatility_vs_frame_turnover.png
    figures/12_integrated_sfp_dynamics/frame_turnover_vs_sentiment_volatility.png
    figures/12_integrated_sfp_dynamics/integrated_sfp_map.png
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STANDING_PATH = (
    PROJECT_ROOT
    / "results"
    / "05_relative_standing"
    / "standing_monthly_relative_label_based.csv"
)
SENTIMENT_PATH = (
    PROJECT_ROOT
    / "results"
    / "10_sentiment_baseline"
    / "sentiment_monthly_agg.csv"
)
FRAMES_OVER_TIME_PATH = (
    PROJECT_ROOT
    / "results"
    / "09_frame_evolution"
    / "frames_over_time_all.csv"
)
TOP_FRAME_PATH = (
    PROJECT_ROOT
    / "results"
    / "09_frame_evolution"
    / "top_frame_over_time_all.csv"
)
TURNOVER_PATH = (
    PROJECT_ROOT
    / "results"
    / "09_frame_evolution"
    / "frame_turnover_rate_by_narrative.csv"
)
TYPOLOGY_PATH = (
    PROJECT_ROOT
    / "results"
    / "08_frame_architecture_types"
    / "frame_architecture_typology.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "12_integrated_sfp_dynamics"
FIGURE_DIR = PROJECT_ROOT / "figures" / "12_integrated_sfp_dynamics"

MASTER_PATH = RESULTS_DIR / "sfp_master_monthly.csv"
TRANSITION_EVENTS_PATH = RESULTS_DIR / "dominant_frame_transition_events.csv"
INTEGRATED_METRICS_PATH = RESULTS_DIR / "sfp_integrated_metrics.csv"
CORRELATION_MATRIX_PATH = RESULTS_DIR / "sfp_correlation_matrix.csv"

SENTIMENT_FRAME_GRID_PATH = FIGURE_DIR / "sentiment_frame_coupling_grid.png"
EVENT_CENTERED_FIGURE_PATH = FIGURE_DIR / "event_centered_sentiment_change.png"
INTEGRATED_TIMELINE_PATH = FIGURE_DIR / "integrated_sfp_timeline.png"
PHASE_SPACE_PATH = FIGURE_DIR / "sfp_phase_space.png"
STANDING_TURNOVER_PATH = FIGURE_DIR / "standing_volatility_vs_frame_turnover.png"
TURNOVER_SENTIMENT_PATH = FIGURE_DIR / "frame_turnover_vs_sentiment_volatility.png"
INTEGRATED_MAP_PATH = FIGURE_DIR / "integrated_sfp_map.png"


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    slug = str(value).strip()
    slug = slug.replace("/", "_")
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")

    return slug or "unknown"


def month_start(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    return dates.dt.to_period("M").dt.to_timestamp()


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mean = numeric.mean()
    std = numeric.std(ddof=0)

    if std == 0 or np.isnan(std):
        return pd.Series([np.nan] * len(numeric), index=numeric.index)

    return (numeric - mean) / std


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_standing(path: Path) -> pd.DataFrame:
    require_file(path, "standing monthly file")

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
        raise ValueError(f"Standing file missing columns: {sorted(missing_columns)}")

    output = df[["narrative", "year_month"]].copy()
    output["month"] = month_start(output["year_month"])
    output["narrative_slug"] = output["narrative"].apply(slugify)
    output["standing_count"] = pd.to_numeric(df["absolute_standing"], errors="coerce")
    output["standing_share"] = pd.to_numeric(df["relative_share"], errors="coerce")
    output["standing_share_percent"] = pd.to_numeric(
        df["relative_share_percent"],
        errors="coerce",
    )

    return output.dropna(subset=["month"])


def load_sentiment(path: Path) -> pd.DataFrame:
    require_file(path, "sentiment monthly file")

    df = pd.read_csv(path, low_memory=False)

    required_columns = {"narrative", "year_month", "avg_valence"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Sentiment file missing columns: {sorted(missing_columns)}")

    keep_columns = ["narrative", "year_month", "avg_valence"]

    for optional_column in [
        "tweet_count",
        "avg_intensity",
        "negative_ratio_percent",
        "neutral_ratio_percent",
        "positive_ratio_percent",
    ]:
        if optional_column in df.columns:
            keep_columns.append(optional_column)

    output = df[keep_columns].copy()
    output["month"] = month_start(output["year_month"])
    output["narrative_slug"] = output["narrative"].apply(slugify)
    output["avg_valence"] = pd.to_numeric(output["avg_valence"], errors="coerce")

    return output.dropna(subset=["month"])


def load_top_frame(path: Path) -> pd.DataFrame:
    require_file(path, "top frame over time file")

    df = pd.read_csv(path, low_memory=False)

    required_columns = {"narrative_slug", "month", "frame", "share"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Top frame file missing columns: {sorted(missing_columns)}")

    output = df.copy()
    output["month"] = month_start(output["month"])
    output["dominant_frame"] = output["frame"].astype(str)
    output["dominant_frame_share"] = pd.to_numeric(output["share"], errors="coerce")
    output["narrative_slug"] = output["narrative_slug"].astype(str)

    if "narrative_label" not in output.columns:
        output["narrative_label"] = output["narrative_slug"]

    keep_columns = [
        "narrative_slug",
        "narrative_label",
        "month",
        "dominant_frame",
        "dominant_frame_share",
    ]

    if "frame_label" in output.columns:
        output["dominant_frame_label"] = output["frame_label"].astype(str)
        keep_columns.append("dominant_frame_label")

    return output[keep_columns].dropna(subset=["month"])


def load_frames_over_time(path: Path) -> pd.DataFrame:
    require_file(path, "frames over time file")

    df = pd.read_csv(path, low_memory=False)

    required_columns = {"narrative_slug", "month", "frame", "share"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Frames-over-time file missing columns: {sorted(missing_columns)}")

    output = df.copy()
    output["month"] = month_start(output["month"])
    output["narrative_slug"] = output["narrative_slug"].astype(str)
    output["frame"] = output["frame"].astype(str)
    output["frame_share"] = pd.to_numeric(output["share"], errors="coerce")

    return output.dropna(subset=["month"])


def load_turnover(path: Path) -> pd.DataFrame:
    require_file(path, "frame turnover file")

    df = pd.read_csv(path, low_memory=False)

    required_columns = {"narrative_slug", "turnover_rate"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Turnover file missing columns: {sorted(missing_columns)}")

    output = df[["narrative_slug", "turnover_rate"]].copy()
    output["narrative_slug"] = output["narrative_slug"].astype(str)
    output["frame_turnover_rate"] = pd.to_numeric(
        output["turnover_rate"],
        errors="coerce",
    )
    output = output.drop(columns=["turnover_rate"])

    return output


def load_typology(path: Path) -> pd.DataFrame:
    require_file(path, "frame architecture typology file")

    df = pd.read_csv(path, low_memory=False)

    required_columns = {"narrative", "effective_frames", "architecture_type"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Typology file missing columns: {sorted(missing_columns)}")

    output = df.copy()
    output["narrative_slug"] = output["narrative"].apply(slugify)
    output["effective_frames"] = pd.to_numeric(
        output["effective_frames"],
        errors="coerce",
    )

    keep_columns = [
        "narrative_slug",
        "effective_frames",
        "architecture_type",
    ]

    if "top1_share_percent" in output.columns:
        keep_columns.append("top1_share_percent")

    return output[keep_columns].drop_duplicates("narrative_slug")


def compute_dominance_index(frames: pd.DataFrame) -> pd.DataFrame:
    sorted_frames = frames.sort_values(
        ["narrative_slug", "month", "frame_share"],
        ascending=[True, True, False],
    )

    top2 = sorted_frames.groupby(["narrative_slug", "month"]).head(2)
    shares = (
        top2.groupby(["narrative_slug", "month"])["frame_share"]
        .apply(list)
        .reset_index()
    )

    def calculate_index(values: list[float]) -> float:
        if not values:
            return np.nan
        if len(values) == 1:
            return float(values[0])
        return float(values[0] - values[1])

    shares["dominance_index"] = shares["frame_share"].apply(calculate_index)

    return shares.drop(columns=["frame_share"])


def build_master_monthly(
    standing: pd.DataFrame,
    sentiment: pd.DataFrame,
    top_frame: pd.DataFrame,
    frames: pd.DataFrame,
) -> pd.DataFrame:
    dominance_index = compute_dominance_index(frames)

    standing_keep = standing[
        [
            "narrative_slug",
            "month",
            "standing_count",
            "standing_share",
            "standing_share_percent",
        ]
    ]

    sentiment_keep = sentiment[
        [
            column
            for column in sentiment.columns
            if column not in ["narrative", "year_month"]
        ]
    ]

    master = standing_keep.merge(
        sentiment_keep,
        on=["narrative_slug", "month"],
        how="outer",
    )

    master = master.merge(
        top_frame,
        on=["narrative_slug", "month"],
        how="outer",
    )

    master = master.merge(
        dominance_index,
        on=["narrative_slug", "month"],
        how="left",
    )

    master = master.sort_values(["narrative_slug", "month"]).reset_index(drop=True)

    if "narrative_label" not in master.columns:
        master["narrative_label"] = master["narrative_slug"]

    master["year_month"] = master["month"].dt.strftime("%Y-%m")
    master["standing_z"] = master.groupby("narrative_slug")["standing_share"].transform(zscore)
    master["sentiment_z"] = master.groupby("narrative_slug")["avg_valence"].transform(zscore)

    master["previous_dominant_frame"] = master.groupby("narrative_slug")[
        "dominant_frame"
    ].shift(1)

    master["frame_changed"] = (
        master["dominant_frame"].notna()
        & master["previous_dominant_frame"].notna()
        & (master["dominant_frame"] != master["previous_dominant_frame"])
    )

    return master


def detect_frame_transition_events(master: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    events = []

    for narrative_slug, group in master.groupby("narrative_slug"):
        group = group.sort_values("month").reset_index(drop=True)

        transition_rows = group[group["frame_changed"]].copy()

        for index, row in transition_rows.iterrows():
            left_index = index - window
            right_index = index + window

            valence_left = (
                group.loc[left_index, "avg_valence"]
                if left_index >= 0
                else np.nan
            )
            valence_right = (
                group.loc[right_index, "avg_valence"]
                if right_index < len(group)
                else np.nan
            )

            delta_valence = (
                valence_right - valence_left
                if pd.notna(valence_left) and pd.notna(valence_right)
                else np.nan
            )

            events.append(
                {
                    "narrative_slug": narrative_slug,
                    "narrative_label": row.get("narrative_label", narrative_slug),
                    "month": row["month"],
                    "from_frame": row["previous_dominant_frame"],
                    "to_frame": row["dominant_frame"],
                    "dominant_frame_share": row.get("dominant_frame_share", np.nan),
                    "dominance_index": row.get("dominance_index", np.nan),
                    "avg_valence_at_transition": row.get("avg_valence", np.nan),
                    "delta_valence_pm1": delta_valence,
                }
            )

    return pd.DataFrame(events).sort_values(["narrative_slug", "month"]).reset_index(drop=True)


def compute_integrated_metrics(
    master: pd.DataFrame,
    turnover: pd.DataFrame,
    typology: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for narrative_slug, group in master.groupby("narrative_slug"):
        group = group.sort_values("month")

        rows.append(
            {
                "narrative_slug": narrative_slug,
                "narrative_label": group["narrative_label"].dropna().iloc[0]
                if group["narrative_label"].notna().any()
                else narrative_slug,
                "n_months": int(group["month"].notna().sum()),
                "standing_mean": float(group["standing_share"].mean()),
                "standing_volatility": float(group["standing_share"].std(ddof=0)),
                "sentiment_mean": float(group["avg_valence"].mean()),
                "sentiment_volatility": float(group["avg_valence"].std(ddof=0)),
                "dominance_index_mean": float(group["dominance_index"].mean()),
                "dominant_frame_share_mean": float(group["dominant_frame_share"].mean()),
                "frame_change_count": int(group["frame_changed"].sum()),
            }
        )

    metrics = pd.DataFrame(rows)

    metrics = metrics.merge(turnover, on="narrative_slug", how="left")
    metrics = metrics.merge(typology, on="narrative_slug", how="left")

    return metrics.sort_values("narrative_slug").reset_index(drop=True)


def plot_sentiment_frame_coupling(master: pd.DataFrame) -> None:
    narratives = sorted(master["narrative_slug"].dropna().unique().tolist())
    n_columns = 2
    n_rows = int(np.ceil(len(narratives) / n_columns))

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_columns,
        figsize=(18, 3.4 * n_rows),
    )
    axes = np.array(axes).reshape(-1)

    for index, narrative_slug in enumerate(narratives):
        ax = axes[index]
        group = master[master["narrative_slug"] == narrative_slug].sort_values("month")

        ax.plot(group["month"], group["avg_valence"], marker="o", linewidth=1.6)
        ax.axhline(0, linewidth=1, color="black", alpha=0.7)
        ax.set_title(group["narrative_label"].dropna().iloc[0] if group["narrative_label"].notna().any() else narrative_slug)
        ax.set_xlabel("Month")
        ax.set_ylabel("Avg valence")
        ax.grid(True, linestyle="--", alpha=0.3)

        ax2 = ax.twinx()
        dominant_frames = group["dominant_frame"].astype(str).replace("nan", np.nan)
        unique_frames = [frame for frame in dominant_frames.dropna().unique()]
        code_map = {frame: code for code, frame in enumerate(unique_frames)}
        frame_codes = dominant_frames.map(code_map)

        ax2.plot(group["month"], frame_codes, linestyle="--", marker=".", color="black", alpha=0.7)
        ax2.set_ylabel("Dominant frame code")

    for index in range(len(narratives), len(axes)):
        axes[index].set_axis_off()

    fig.suptitle(
        "Sentiment-Frame Coupling Over Time",
        y=0.995,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(SENTIMENT_FRAME_GRID_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_event_centered_sentiment_change(events: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_title(
        "Event-Centered Sentiment Change Around Dominant-Frame Transitions",
        fontweight="bold",
    )
    ax.set_xlabel("Narrative")
    ax.set_ylabel("Delta valence: t+1 minus t-1")
    ax.axhline(0, linewidth=1, color="black", alpha=0.7)

    valid_events = events.dropna(subset=["delta_valence_pm1"]).copy()

    if valid_events.empty:
        ax.text(
            0.5,
            0.5,
            "No valid transition events with +/-1-month context.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig.savefig(EVENT_CENTERED_FIGURE_PATH, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    narratives = sorted(valid_events["narrative_label"].unique().tolist())
    x_map = {narrative: index for index, narrative in enumerate(narratives)}
    rng = np.random.default_rng(42)

    x_values = valid_events["narrative_label"].map(x_map).astype(float).to_numpy()
    jitter = rng.uniform(-0.09, 0.09, size=len(x_values))

    ax.scatter(
        x_values + jitter,
        valid_events["delta_valence_pm1"],
        alpha=0.75,
    )

    means = valid_events.groupby("narrative_label")["delta_valence_pm1"].mean().reindex(narratives)
    ax.plot(range(len(narratives)), means.values, marker="o", linestyle="-", color="black")

    ax.set_xticks(range(len(narratives)))
    ax.set_xticklabels(narratives, rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(EVENT_CENTERED_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_frame_color_map(master: pd.DataFrame) -> dict[str, object]:
    frame_labels = sorted(master["dominant_frame"].dropna().astype(str).unique().tolist())
    color_map = plt.get_cmap("tab20")

    return {
        label: color_map(index % 20)
        for index, label in enumerate(frame_labels)
    }


def plot_integrated_timeline(master: pd.DataFrame) -> None:
    narratives = sorted(master["narrative_slug"].dropna().unique().tolist())
    n_columns = 2
    n_rows = int(np.ceil(len(narratives) / n_columns))
    frame_colors = get_frame_color_map(master)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_columns,
        figsize=(18, 4.3 * n_rows),
    )
    axes = np.array(axes).reshape(-1)

    for index, narrative_slug in enumerate(narratives):
        ax = axes[index]
        group = master[master["narrative_slug"] == narrative_slug].sort_values("month")

        for _, row in group.dropna(subset=["month", "dominant_frame"]).iterrows():
            start = row["month"]
            end = row["month"] + pd.offsets.MonthBegin(1)
            ax.axvspan(
                start,
                end,
                alpha=0.12,
                color=frame_colors.get(str(row["dominant_frame"])),
            )

        ax.plot(group["month"], group["standing_z"], linewidth=1.8, label="Standing (z)")
        ax.plot(group["month"], group["sentiment_z"], linewidth=1.4, linestyle="--", label="Sentiment (z)")
        ax.axhline(0, linewidth=1, color="black", alpha=0.7)

        ax.set_title(group["narrative_label"].dropna().iloc[0] if group["narrative_label"].notna().any() else narrative_slug)
        ax.set_ylabel("Z-score")
        ax.grid(True, linestyle="--", alpha=0.3)

        visible_frames = (
            group["dominant_frame"]
            .dropna()
            .astype(str)
            .value_counts()
            .head(5)
            .index
            .tolist()
        )

        if visible_frames:
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    color=frame_colors.get(frame),
                    linewidth=8,
                    alpha=0.3,
                )
                for frame in visible_frames
            ]
            ax.legend(handles, visible_frames, title="Dominant frames", fontsize=8, title_fontsize=9)

    for index in range(len(narratives), len(axes)):
        axes[index].set_axis_off()

    fig.suptitle(
        "Integrated SFP Timeline",
        y=0.995,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(INTEGRATED_TIMELINE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_phase_space(master: pd.DataFrame) -> None:
    narratives = sorted(master["narrative_slug"].dropna().unique().tolist())
    n_columns = 2
    n_rows = int(np.ceil(len(narratives) / n_columns))
    frame_colors = get_frame_color_map(master)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_columns,
        figsize=(18, 4.3 * n_rows),
    )
    axes = np.array(axes).reshape(-1)

    for index, narrative_slug in enumerate(narratives):
        ax = axes[index]
        group = master[master["narrative_slug"] == narrative_slug].dropna(
            subset=["standing_z", "sentiment_z"]
        )

        if group.empty:
            ax.set_axis_off()
            continue

        for frame, frame_group in group.groupby("dominant_frame"):
            ax.scatter(
                frame_group["standing_z"],
                frame_group["sentiment_z"],
                s=28,
                alpha=0.75,
                color=frame_colors.get(str(frame)),
                label=str(frame),
            )

        ax.axhline(0, linewidth=1, color="black", alpha=0.7)
        ax.axvline(0, linewidth=1, color="black", alpha=0.7)
        ax.set_title(group["narrative_label"].dropna().iloc[0] if group["narrative_label"].notna().any() else narrative_slug)
        ax.set_xlabel("Standing (z)")
        ax.set_ylabel("Sentiment (z)")
        ax.grid(True, linestyle="--", alpha=0.3)

    for index in range(len(narratives), len(axes)):
        axes[index].set_axis_off()

    fig.suptitle(
        "SFP Phase Space",
        y=0.995,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(PHASE_SPACE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def annotate_points(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    label_column: str,
) -> None:
    for _, row in df.iterrows():
        if pd.isna(row[x_column]) or pd.isna(row[y_column]):
            continue

        ax.text(
            row[x_column],
            row[y_column],
            str(row[label_column]),
            fontsize=8,
        )


def plot_metric_scatter(
    metrics: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: Path,
    x_label: str,
    y_label: str,
    title: str,
    bubble_size_column: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    if bubble_size_column is not None and bubble_size_column in metrics.columns:
        size_values = pd.to_numeric(metrics[bubble_size_column], errors="coerce")
        size_values = size_values.fillna(size_values.min() if pd.notna(size_values.min()) else 0)
        if size_values.max() > size_values.min():
            sizes = (size_values - size_values.min()) / (size_values.max() - size_values.min()) * 1800 + 120
        else:
            sizes = 240
    else:
        sizes = 120

    ax.scatter(metrics[x_column], metrics[y_column], s=sizes, alpha=0.75)
    annotate_points(ax, metrics, x_column, y_column, "narrative_label")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metric_figures(metrics: pd.DataFrame) -> None:
    plot_metric_scatter(
        metrics,
        x_column="standing_volatility",
        y_column="frame_turnover_rate",
        output_path=STANDING_TURNOVER_PATH,
        x_label="Standing volatility",
        y_label="Frame turnover rate",
        title="Standing Volatility vs. Frame Turnover",
    )

    plot_metric_scatter(
        metrics,
        x_column="frame_turnover_rate",
        y_column="sentiment_volatility",
        output_path=TURNOVER_SENTIMENT_PATH,
        x_label="Frame turnover rate",
        y_label="Sentiment volatility",
        title="Frame Turnover vs. Sentiment Volatility",
    )

    plot_metric_scatter(
        metrics,
        x_column="frame_turnover_rate",
        y_column="sentiment_volatility",
        output_path=INTEGRATED_MAP_PATH,
        x_label="Frame turnover rate",
        y_label="Sentiment volatility",
        title="Integrated SFP Map",
        bubble_size_column="standing_mean",
    )


def save_correlation_matrix(metrics: pd.DataFrame) -> None:
    columns = [
        "standing_volatility",
        "frame_turnover_rate",
        "sentiment_volatility",
        "effective_frames",
        "dominance_index_mean",
    ]

    available_columns = [column for column in columns if column in metrics.columns]
    correlation = metrics[available_columns].corr()
    correlation.to_csv(CORRELATION_MATRIX_PATH, encoding="utf-8")


def main() -> None:
    ensure_output_dirs()

    standing = load_standing(STANDING_PATH)
    sentiment = load_sentiment(SENTIMENT_PATH)
    frames = load_frames_over_time(FRAMES_OVER_TIME_PATH)
    top_frame = load_top_frame(TOP_FRAME_PATH)
    turnover = load_turnover(TURNOVER_PATH)
    typology = load_typology(TYPOLOGY_PATH)

    master = build_master_monthly(
        standing=standing,
        sentiment=sentiment,
        top_frame=top_frame,
        frames=frames,
    )
    master.to_csv(MASTER_PATH, index=False, encoding="utf-8")

    transition_events = detect_frame_transition_events(master, window=1)
    transition_events.to_csv(TRANSITION_EVENTS_PATH, index=False, encoding="utf-8")

    metrics = compute_integrated_metrics(master, turnover, typology)
    metrics.to_csv(INTEGRATED_METRICS_PATH, index=False, encoding="utf-8")
    save_correlation_matrix(metrics)

    plot_sentiment_frame_coupling(master)
    plot_event_centered_sentiment_change(transition_events)
    plot_integrated_timeline(master)
    plot_phase_space(master)
    save_metric_figures(metrics)

    print("\nSaved outputs")
    print(f"Master table:         {MASTER_PATH}")
    print(f"Transition events:    {TRANSITION_EVENTS_PATH}")
    print(f"Integrated metrics:   {INTEGRATED_METRICS_PATH}")
    print(f"Correlation matrix:   {CORRELATION_MATRIX_PATH}")
    print(f"Figure:               {SENTIMENT_FRAME_GRID_PATH}")
    print(f"Figure:               {EVENT_CENTERED_FIGURE_PATH}")
    print(f"Figure:               {INTEGRATED_TIMELINE_PATH}")
    print(f"Figure:               {PHASE_SPACE_PATH}")
    print(f"Figure:               {STANDING_TURNOVER_PATH}")
    print(f"Figure:               {TURNOVER_SENTIMENT_PATH}")
    print(f"Figure:               {INTEGRATED_MAP_PATH}")

    print("\nIntegrated metrics preview")
    print(metrics.head(12).to_string(index=False))


if __name__ == "__main__":
    main()