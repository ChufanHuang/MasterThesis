#!/usr/bin/env python3
"""
Compute and visualize frame evolution over time.

Thesis section:
    4.3.3 Temporal Evolution of Frames

Inputs:
    results/07_bertopic_framing/*_tweets_with_frames.csv
    results/07_bertopic_framing/*_frame_info.csv
    results/08_frame_architecture_types/frame_architecture_typology.csv

Outputs:
    results/09_frame_evolution/frames_over_time_all.csv
    results/09_frame_evolution/top_frame_over_time_all.csv
    results/09_frame_evolution/frames_monthly_indices_all.csv
    results/09_frame_evolution/frame_turnover_rate_by_narrative.csv
    results/09_frame_evolution/frame_turnover_rate_by_type.csv
    results/09_frame_evolution/by_narrative/{narrative}/frames_over_time.csv
    results/09_frame_evolution/by_narrative/{narrative}/top_frame_over_time.csv
    results/09_frame_evolution/by_narrative/{narrative}/frames_monthly_indices.csv
    figures/09_frame_evolution/{narrative}_top_frames_over_time.png
    figures/09_frame_evolution/{narrative}_top1_share_with_replacements.png
    figures/09_frame_evolution/all_top_frames_grid.png
    figures/09_frame_evolution/all_top1_replacement_grid.png
    figures/09_frame_evolution/frame_turnover_rate.png
    figures/09_frame_evolution/turnover_vs_effective_frames.png
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FRAMING_INPUT_DIR = PROJECT_ROOT / "results" / "07_bertopic_framing"
TYPOLOGY_PATH = (
    PROJECT_ROOT
    / "results"
    / "08_frame_architecture_types"
    / "frame_architecture_typology.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "09_frame_evolution"
BY_NARRATIVE_DIR = RESULTS_DIR / "by_narrative"
FIGURE_DIR = PROJECT_ROOT / "figures" / "09_frame_evolution"

FRAMES_OVER_TIME_ALL_PATH = RESULTS_DIR / "frames_over_time_all.csv"
TOP_FRAME_OVER_TIME_ALL_PATH = RESULTS_DIR / "top_frame_over_time_all.csv"
MONTHLY_INDICES_ALL_PATH = RESULTS_DIR / "frames_monthly_indices_all.csv"
TURNOVER_BY_NARRATIVE_PATH = RESULTS_DIR / "frame_turnover_rate_by_narrative.csv"
TURNOVER_BY_TYPE_PATH = RESULTS_DIR / "frame_turnover_rate_by_type.csv"

ALL_TOP_FRAMES_GRID_PATH = FIGURE_DIR / "all_top_frames_grid.png"
ALL_TOP1_REPLACEMENT_GRID_PATH = FIGURE_DIR / "all_top1_replacement_grid.png"
TURNOVER_FIGURE_PATH = FIGURE_DIR / "frame_turnover_rate.png"
TURNOVER_VS_EFFECTIVE_FRAMES_PATH = FIGURE_DIR / "turnover_vs_effective_frames.png"

TWEETS_WITH_FRAMES_GLOB = "*_tweets_with_frames.csv"

REMOVE_OUTLIERS = True
MIN_MONTH_TOTAL = 1
TOP_K_FRAMES = 5
MIN_MONTHS_TO_PLOT = 6
DATE_FORMAT = "%Y-%m"

NARRATIVE_ORDER = [
    "Flat_Earth",
    "Geoengineering",
    "Chemtrails",
    "Fluoride_Cancer_Misc",
    "5G",
    "GMO_Monsanto_Glyphosate",
    "Directed_Energy_Weapons",
    "NASA___Space_Hoax",
    "Anti_vaccination",
    "HAARP",
]

NARRATIVE_LABEL_MAP = {
    "NASA_Space_Hoax": "NASA / Space Hoax",
    "NASA___Space_Hoax": "NASA / Space Hoax",
    "GMO_Monsanto_Glyphosate": "GMO / Monsanto / Glyphosate",
    "Fluoride_Cancer_Misc": "Fluoride / Cancer (Misc.)",
    "Directed_Energy_Weapons": "Directed Energy Weapons",
    "Anti_vaccination": "Anti-vaccination",
}


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BY_NARRATIVE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    slug = str(value).strip()
    slug = slug.replace("/", "_")
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "", slug)

    return slug or "unknown"


def month_to_period(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, utc=True, errors="coerce")
    dates = dates.dt.tz_convert(None)
    return dates.dt.to_period("M")


def month_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=DATE_FORMAT, errors="coerce")


def get_narrative_label(narrative_slug: str) -> str:
    return NARRATIVE_LABEL_MAP.get(narrative_slug, narrative_slug)


def load_frame_labels(narrative_slug: str) -> dict[int, str]:
    frame_info_path = FRAMING_INPUT_DIR / f"{narrative_slug}_frame_info.csv"

    if not frame_info_path.exists():
        return {}

    frame_info = pd.read_csv(frame_info_path, low_memory=False)

    if "Topic" not in frame_info.columns:
        return {}

    label_column = None
    for candidate in ["Name", "Representation", "Label"]:
        if candidate in frame_info.columns:
            label_column = candidate
            break

    if label_column is None:
        return {}

    frame_info = frame_info.dropna(subset=["Topic"]).copy()
    frame_info["Topic"] = pd.to_numeric(frame_info["Topic"], errors="coerce")
    frame_info = frame_info.dropna(subset=["Topic"]).copy()
    frame_info["Topic"] = frame_info["Topic"].astype(int)
    frame_info[label_column] = frame_info[label_column].astype(str)

    return dict(zip(frame_info["Topic"], frame_info[label_column]))


def compute_monthly_frame_distribution(
    tweets_with_frames: pd.DataFrame,
    narrative_slug: str,
) -> pd.DataFrame:
    frame_column = None

    if "frame" in tweets_with_frames.columns:
        frame_column = "frame"
    elif "Topic" in tweets_with_frames.columns:
        frame_column = "Topic"

    if frame_column is None:
        raise ValueError("Missing frame column. Expected 'frame' or 'Topic'.")

    if "created_at" not in tweets_with_frames.columns:
        raise ValueError("Missing required column: created_at")

    df = tweets_with_frames.copy()
    df["month"] = month_to_period(df["created_at"])
    df = df.dropna(subset=["month"]).copy()

    df[frame_column] = pd.to_numeric(df[frame_column], errors="coerce")
    df = df.dropna(subset=[frame_column]).copy()
    df[frame_column] = df[frame_column].astype(int)

    if REMOVE_OUTLIERS:
        df = df[df[frame_column] != -1].copy()

    if df.empty:
        return pd.DataFrame()

    monthly = (
        df.groupby(["month", frame_column])
        .size()
        .reset_index(name="count")
        .rename(columns={frame_column: "frame"})
    )

    month_min = monthly["month"].min()
    month_max = monthly["month"].max()
    all_months = pd.period_range(month_min, month_max, freq="M")
    all_frames = monthly["frame"].unique()

    full_index = pd.MultiIndex.from_product(
        [all_months, all_frames],
        names=["month", "frame"],
    )

    monthly = (
        monthly.set_index(["month", "frame"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    totals = monthly.groupby("month")["count"].sum().reset_index(name="month_total")
    monthly = monthly.merge(totals, on="month", how="left")
    monthly = monthly[monthly["month_total"] >= MIN_MONTH_TOTAL].copy()

    monthly["share"] = np.where(
        monthly["month_total"] > 0,
        monthly["count"] / monthly["month_total"],
        0.0,
    )

    monthly["narrative_slug"] = narrative_slug
    monthly["narrative_label"] = get_narrative_label(narrative_slug)

    frame_labels = load_frame_labels(narrative_slug)
    if frame_labels:
        monthly["frame_label"] = monthly["frame"].map(frame_labels)

    monthly["month"] = monthly["month"].astype(str)

    return monthly


def compute_top_frame_over_time(frames_over_time: pd.DataFrame) -> pd.DataFrame:
    top_frame = (
        frames_over_time.sort_values(
            ["month", "share", "count"],
            ascending=[True, False, False],
        )
        .groupby(["narrative_slug", "month"], as_index=False)
        .head(1)
        .copy()
    )

    keep_columns = [
        "narrative_slug",
        "narrative_label",
        "month",
        "frame",
        "share",
        "count",
    ]

    if "frame_label" in top_frame.columns:
        keep_columns.append("frame_label")

    return top_frame[keep_columns].reset_index(drop=True)


def compute_monthly_indices(frames_over_time: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (slug, label, month), group in frames_over_time.groupby(
        ["narrative_slug", "narrative_label", "month"]
    ):
        shares = group["share"].to_numpy(dtype=float)

        if shares.sum() <= 0:
            continue

        shares = shares / shares.sum()
        sorted_shares = np.sort(shares)[::-1]

        top1_share = float(sorted_shares[0])
        top3_share = (
            float(sorted_shares[:3].sum())
            if len(sorted_shares) >= 3
            else float(sorted_shares.sum())
        )

        hhi = float(np.sum(shares**2))
        effective_frames = float(1.0 / hhi) if hhi > 0 else np.nan
        entropy = float(-np.sum(shares * np.log(shares + 1e-12)))

        top_row = group.sort_values(
            ["share", "count"],
            ascending=[False, False],
        ).iloc[0]

        rows.append(
            {
                "narrative_slug": slug,
                "narrative_label": label,
                "month": month,
                "top1_share": top1_share,
                "top3_share": top3_share,
                "hhi": hhi,
                "effective_frames": effective_frames,
                "entropy": entropy,
                "top1_frame": int(top_row["frame"]),
            }
        )

    indices = (
        pd.DataFrame(rows)
        .sort_values(["narrative_slug", "month"])
        .reset_index(drop=True)
    )

    if indices.empty:
        return indices

    indices["top1_changed"] = indices["top1_frame"].ne(
        indices.groupby("narrative_slug")["top1_frame"].shift(1)
    )
    indices.loc[indices.groupby("narrative_slug").head(1).index, "top1_changed"] = False

    indices["top1_share_percent"] = (indices["top1_share"] * 100).round(2)
    indices["top3_share_percent"] = (indices["top3_share"] * 100).round(2)

    return indices


def pick_top_frames(frames_over_time: pd.DataFrame, top_k: int) -> list[int]:
    top_frames = (
        frames_over_time.groupby("frame")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_k)
        .index
        .tolist()
    )

    return [int(frame) for frame in top_frames]


def build_frame_labels(frames_over_time: pd.DataFrame) -> dict[int, str]:
    if "frame_label" not in frames_over_time.columns:
        return {}

    labels = (
        frames_over_time[["frame", "frame_label"]]
        .dropna()
        .drop_duplicates()
    )

    output = {}

    for _, row in labels.iterrows():
        try:
            output[int(row["frame"])] = str(row["frame_label"])
        except ValueError:
            continue

    return output


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_top_frames_figure(
    narrative_slug: str,
    frames_over_time: pd.DataFrame,
) -> Path | None:
    df = frames_over_time.copy()
    required_columns = {"month", "frame", "count", "share"}

    if not required_columns <= set(df.columns):
        return None

    df["month_dt"] = month_to_datetime(df["month"])
    df = df.dropna(subset=["month_dt"]).sort_values("month_dt")

    if df["month_dt"].nunique() < MIN_MONTHS_TO_PLOT:
        print(f"Skipping top-frame figure for {narrative_slug}: too few months")
        return None

    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df["share"] = pd.to_numeric(df["share"], errors="coerce")
    df = df.dropna(subset=["frame", "count", "share"]).copy()
    df["frame"] = df["frame"].astype(int)

    top_frames = pick_top_frames(df, TOP_K_FRAMES)
    frame_labels = build_frame_labels(df)

    pivot = (
        df[df["frame"].isin(top_frames)]
        .pivot_table(
            index="month_dt",
            columns="frame",
            values="share",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    for frame_id in pivot.columns:
        if frame_id in frame_labels:
            label = f"F{frame_id}: {frame_labels[frame_id]}"
        else:
            label = f"Frame {frame_id}"

        ax.plot(pivot.index, pivot[frame_id] * 100, label=label, linewidth=2)

    narrative_label = get_narrative_label(narrative_slug)

    ax.set_title(
        f"Top {TOP_K_FRAMES} Frames Over Time: {narrative_label}",
        fontweight="bold",
    )
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Monthly frame share within narrative (%)")
    style_axis(ax)
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    output_path = FIGURE_DIR / f"{narrative_slug}_top_frames_over_time.png"
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_top1_replacement_figure(
    narrative_slug: str,
    top_frame_over_time: pd.DataFrame,
) -> Path | None:
    df = top_frame_over_time.copy()
    required_columns = {"month", "frame", "share"}

    if not required_columns <= set(df.columns):
        return None

    df["month_dt"] = month_to_datetime(df["month"])
    df = df.dropna(subset=["month_dt"]).sort_values("month_dt")

    if df["month_dt"].nunique() < MIN_MONTHS_TO_PLOT:
        print(f"Skipping replacement figure for {narrative_slug}: too few months")
        return None

    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["share"] = pd.to_numeric(df["share"], errors="coerce")
    df = df.dropna(subset=["frame", "share"]).copy()
    df["frame"] = df["frame"].astype(int)

    df["previous_frame"] = df["frame"].shift(1)
    df["changed"] = df["frame"].ne(df["previous_frame"])
    df.loc[df.index.min(), "changed"] = False

    fig, ax = plt.subplots(figsize=(12, 3.5))

    ax.plot(
        df["month_dt"],
        df["share"] * 100,
        linewidth=2.5,
        label="Top-1 frame share",
    )

    for month in df.loc[df["changed"], "month_dt"]:
        ax.axvline(month, linestyle="--", linewidth=1.2, alpha=0.6)

    if not df.empty:
        peak_index = df["share"].idxmax()
        peak_month = df.loc[peak_index, "month_dt"]
        peak_value = float(df.loc[peak_index, "share"] * 100)

        ax.annotate(
            f"Peak {peak_month.strftime('%Y-%m')}: {peak_value:.1f}%",
            xy=(peak_month, peak_value),
            xytext=(peak_month, peak_value + 8),
            arrowprops={"arrowstyle": "->", "linewidth": 1},
            fontsize=9,
        )

    narrative_label = get_narrative_label(narrative_slug)

    ax.set_title(
        f"Top-1 Frame Share and Replacement Points: {narrative_label}",
        fontweight="bold",
    )
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Top-1 share (%)")
    style_axis(ax)

    output_path = FIGURE_DIR / f"{narrative_slug}_top1_share_with_replacements.png"
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def make_image_grid(image_paths: list[Path], output_path: Path, columns: int = 2) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths if path.exists()]

    if not images:
        print(f"Skipping grid: no images for {output_path.name}")
        return

    rows = (len(images) + columns - 1) // columns
    column_widths = [0] * columns
    row_heights = [0] * rows

    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        column_widths[column] = max(column_widths[column], image.width)
        row_heights[row] = max(row_heights[row], image.height)

    canvas = Image.new(
        "RGB",
        (sum(column_widths), sum(row_heights)),
        color=(255, 255, 255),
    )

    y = 0
    for row in range(rows):
        x = 0
        for column in range(columns):
            index = row * columns + column
            if index >= len(images):
                break

            canvas.paste(images[index], (x, y))
            x += column_widths[column]

        y += row_heights[row]

    canvas.save(output_path, format="PNG")


def count_replacements(frames: pd.Series) -> int:
    sequence = frames.astype("Int64")
    changed = sequence.ne(sequence.shift(1))

    if len(changed) <= 1:
        return 0

    return int(changed.iloc[1:].sum())


def load_typology() -> pd.DataFrame | None:
    if not TYPOLOGY_PATH.exists():
        print(f"Typology file not found, skipping type merge: {TYPOLOGY_PATH}")
        return None

    typology = pd.read_csv(TYPOLOGY_PATH)

    required_columns = {"narrative", "architecture_type", "effective_frames"}
    missing_columns = required_columns - set(typology.columns)

    if missing_columns:
        print(f"Typology file missing columns: {sorted(missing_columns)}")
        return None

    typology = typology.copy()
    typology["narrative_slug"] = typology["narrative"].apply(slugify)

    return typology[
        [
            "narrative_slug",
            "architecture_type",
            "effective_frames",
            "top1_share_percent",
        ]
    ].drop_duplicates()


def compute_turnover_table(
    top_frame_over_time_all: pd.DataFrame,
    typology: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []

    for narrative_slug, group in top_frame_over_time_all.groupby("narrative_slug"):
        group = group.copy()
        group["month_dt"] = month_to_datetime(group["month"])
        group = group.dropna(subset=["month_dt"]).sort_values("month_dt")
        group = group.drop_duplicates("month_dt", keep="first")

        n_months = int(group["month_dt"].nunique())

        if n_months <= 1:
            n_replacements = 0
            turnover_rate = 0.0
        else:
            n_replacements = count_replacements(group["frame"])
            turnover_rate = n_replacements / (n_months - 1)

        share = pd.to_numeric(group["share"], errors="coerce")

        rows.append(
            {
                "narrative_slug": narrative_slug,
                "narrative_label": get_narrative_label(narrative_slug),
                "n_months": n_months,
                "n_top1_replacements": n_replacements,
                "turnover_rate": round(float(turnover_rate), 4),
                "top1_share_mean_percent": round(float(share.mean() * 100), 2)
                if share.notna().any()
                else np.nan,
                "top1_share_median_percent": round(float(share.median() * 100), 2)
                if share.notna().any()
                else np.nan,
                "top1_share_sd_percent": round(float(share.std() * 100), 2)
                if share.notna().any()
                else np.nan,
            }
        )

    turnover = pd.DataFrame(rows)

    if typology is not None:
        turnover = turnover.merge(typology, on="narrative_slug", how="left")

    return (
        turnover.sort_values(
            ["turnover_rate", "n_months"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def build_turnover_by_type(turnover: pd.DataFrame) -> pd.DataFrame:
    if "architecture_type" not in turnover.columns:
        return pd.DataFrame()

    summary = (
        turnover.groupby("architecture_type", as_index=False)
        .agg(
            n_narratives=("narrative_slug", "count"),
            turnover_rate_mean=("turnover_rate", "mean"),
            turnover_rate_median=("turnover_rate", "median"),
            turnover_rate_min=("turnover_rate", "min"),
            turnover_rate_max=("turnover_rate", "max"),
            replacements_sum=("n_top1_replacements", "sum"),
            months_sum=("n_months", "sum"),
        )
        .round(
            {
                "turnover_rate_mean": 4,
                "turnover_rate_median": 4,
                "turnover_rate_min": 4,
                "turnover_rate_max": 4,
            }
        )
        .sort_values("n_narratives", ascending=False)
        .reset_index(drop=True)
    )

    return summary


def save_turnover_bar_figure(turnover: pd.DataFrame) -> None:
    plot_df = turnover.copy()

    order = [
        narrative
        for narrative in NARRATIVE_ORDER
        if narrative in set(plot_df["narrative_slug"])
    ]

    if len(order) < len(plot_df):
        rest = sorted(
            narrative
            for narrative in plot_df["narrative_slug"].unique()
            if narrative not in order
        )
        order += rest

    plot_df["narrative_slug"] = pd.Categorical(
        plot_df["narrative_slug"],
        categories=order,
        ordered=True,
    )
    plot_df = plot_df.sort_values("narrative_slug")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(plot_df["narrative_slug"].astype(str), plot_df["turnover_rate"])

    ax.set_title(
        "Frame Turnover Rate by Narrative",
        fontweight="bold",
    )
    ax.set_xlabel("Narrative")
    ax.set_ylabel("Turnover rate")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(TURNOVER_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_turnover_vs_effective_frames_figure(turnover: pd.DataFrame) -> None:
    required_columns = {"effective_frames", "turnover_rate"}

    if not required_columns <= set(turnover.columns):
        print("Skipping turnover-vs-effective-frames figure: missing typology columns")
        return

    plot_df = turnover.dropna(subset=["effective_frames", "turnover_rate"]).copy()

    if plot_df.empty:
        print("Skipping turnover-vs-effective-frames figure: no merged rows")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    if "top1_share_mean_percent" in plot_df.columns:
        sizes = plot_df["top1_share_mean_percent"].fillna(10) * 8
    else:
        sizes = 80

    ax.scatter(
        plot_df["effective_frames"],
        plot_df["turnover_rate"],
        s=sizes,
        alpha=0.7,
    )

    for _, row in plot_df.iterrows():
        ax.text(
            float(row["effective_frames"]) + 0.15,
            float(row["turnover_rate"]) + 0.01,
            str(row["narrative_label"]),
            fontsize=9,
        )

    ax.set_xlabel("Effective number of frames")
    ax.set_ylabel("Frame turnover rate")
    ax.set_title(
        "Frame Diversity and Turnover Across Narratives",
        fontweight="bold",
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(TURNOVER_VS_EFFECTIVE_FRAMES_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_narrative_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    narrative_slug = path.stem.replace("_tweets_with_frames", "")
    tweets_with_frames = pd.read_csv(path, low_memory=False)

    frames_over_time = compute_monthly_frame_distribution(
        tweets_with_frames=tweets_with_frames,
        narrative_slug=narrative_slug,
    )

    if frames_over_time.empty:
        print(f"Skipping {narrative_slug}: empty frame distribution")
        return None

    top_frame_over_time = compute_top_frame_over_time(frames_over_time)
    monthly_indices = compute_monthly_indices(frames_over_time)

    narrative_output_dir = BY_NARRATIVE_DIR / narrative_slug
    narrative_output_dir.mkdir(parents=True, exist_ok=True)

    frames_over_time.to_csv(
        narrative_output_dir / "frames_over_time.csv",
        index=False,
        encoding="utf-8",
    )
    top_frame_over_time.to_csv(
        narrative_output_dir / "top_frame_over_time.csv",
        index=False,
        encoding="utf-8",
    )
    monthly_indices.to_csv(
        narrative_output_dir / "frames_monthly_indices.csv",
        index=False,
        encoding="utf-8",
    )

    return frames_over_time, top_frame_over_time, monthly_indices


def main() -> None:
    ensure_output_dirs()

    if not FRAMING_INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {FRAMING_INPUT_DIR}. "
            "Run scripts/07_bertopic_framing.py first."
        )

    input_files = sorted(FRAMING_INPUT_DIR.glob(TWEETS_WITH_FRAMES_GLOB))

    if not input_files:
        raise FileNotFoundError(
            f"No files matched: {FRAMING_INPUT_DIR / TWEETS_WITH_FRAMES_GLOB}"
        )

    frames_parts = []
    top_frame_parts = []
    monthly_index_parts = []
    top_frames_figures = []
    replacement_figures = []

    for path in input_files:
        result = process_narrative_file(path)

        if result is None:
            continue

        frames_over_time, top_frame_over_time, monthly_indices = result
        narrative_slug = frames_over_time["narrative_slug"].iloc[0]

        frames_parts.append(frames_over_time)
        top_frame_parts.append(top_frame_over_time)
        monthly_index_parts.append(monthly_indices)

        top_frames_path = save_top_frames_figure(narrative_slug, frames_over_time)
        replacement_path = save_top1_replacement_figure(
            narrative_slug,
            top_frame_over_time,
        )

        if top_frames_path is not None:
            top_frames_figures.append(top_frames_path)

        if replacement_path is not None:
            replacement_figures.append(replacement_path)

    frames_all = pd.concat(frames_parts, ignore_index=True)
    top_frame_all = pd.concat(top_frame_parts, ignore_index=True)
    monthly_indices_all = pd.concat(monthly_index_parts, ignore_index=True)

    frames_all.to_csv(FRAMES_OVER_TIME_ALL_PATH, index=False, encoding="utf-8")
    top_frame_all.to_csv(TOP_FRAME_OVER_TIME_ALL_PATH, index=False, encoding="utf-8")
    monthly_indices_all.to_csv(MONTHLY_INDICES_ALL_PATH, index=False, encoding="utf-8")

    make_image_grid(top_frames_figures, ALL_TOP_FRAMES_GRID_PATH, columns=2)
    make_image_grid(replacement_figures, ALL_TOP1_REPLACEMENT_GRID_PATH, columns=2)

    typology = load_typology()
    turnover = compute_turnover_table(top_frame_all, typology)
    turnover_by_type = build_turnover_by_type(turnover)

    turnover.to_csv(TURNOVER_BY_NARRATIVE_PATH, index=False, encoding="utf-8")

    if not turnover_by_type.empty:
        turnover_by_type.to_csv(TURNOVER_BY_TYPE_PATH, index=False, encoding="utf-8")

    save_turnover_bar_figure(turnover)
    save_turnover_vs_effective_frames_figure(turnover)

    print("\nSaved outputs")
    print(f"Frames over time:       {FRAMES_OVER_TIME_ALL_PATH}")
    print(f"Top frames over time:   {TOP_FRAME_OVER_TIME_ALL_PATH}")
    print(f"Monthly indices:        {MONTHLY_INDICES_ALL_PATH}")
    print(f"Turnover by narrative:  {TURNOVER_BY_NARRATIVE_PATH}")
    print(f"Turnover by type:       {TURNOVER_BY_TYPE_PATH}")
    print(f"Top frames grid:        {ALL_TOP_FRAMES_GRID_PATH}")
    print(f"Replacement grid:       {ALL_TOP1_REPLACEMENT_GRID_PATH}")
    print(f"Turnover figure:        {TURNOVER_FIGURE_PATH}")
    print(f"Turnover vs diversity:  {TURNOVER_VS_EFFECTIVE_FRAMES_PATH}")

    print("\nHighest turnover narratives")
    print(
        turnover.sort_values("turnover_rate", ascending=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()