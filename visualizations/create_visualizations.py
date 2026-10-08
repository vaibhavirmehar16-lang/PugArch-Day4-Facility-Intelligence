"""
Generate publication-quality static visualizations.

Outputs:
    visualizations/bar_cleanliness_by_location.png
    visualizations/bar_complaints_by_water.png
    visualizations/hist_footfall_distribution.png
    visualizations/scatter_odor_vs_cleanliness.png
    visualizations/heatmap_correlation_matrix.png

Usage:
    python visualizations/create_visualizations.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "cleaned_facility_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "visualizations"

sns.set_theme(style="whitegrid", context="notebook")


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["inspection_date"])


def save_figure(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def chart_cleanliness_by_location(df: pd.DataFrame) -> None:
    summary = (
        df.groupby("location")["cleanliness_score"]
        .agg(["mean", "std"])
        .sort_values("mean")
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        summary.index,
        summary["mean"],
        yerr=summary["std"].fillna(0),
        capsize=5,
    )
    ax.set_title("Mean Cleanliness Score by Location")
    ax.set_xlabel("Location")
    ax.set_ylabel("Mean Cleanliness Score")
    ax.set_ylim(0, 10)
    plt.xticks(rotation=20)
    save_figure(fig, "bar_cleanliness_by_location.png")


def chart_complaints_by_water(df: pd.DataFrame) -> None:
    summary = (
        df.groupby("water_availability")["complaints"]
        .sum()
        .reindex(["Yes", "No"])
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(summary.index, summary.values)

    for bar, value in zip(bars, summary.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,.0f}",
            ha="center",
            va="bottom",
        )

    ax.set_title("Total Complaint Volume by Water Availability")
    ax.set_xlabel("Water Availability")
    ax.set_ylabel("Total Complaints")
    save_figure(fig, "bar_complaints_by_water.png")


def chart_footfall_distribution(df: pd.DataFrame) -> None:
    values = df["footfall"].dropna().to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=25, density=True, alpha=0.7)

    # KDE is calculated explicitly with SciPy so the visualization
    # remains robust across Seaborn/Matplotlib versions.
    if len(values) > 1 and np.std(values) > 0:
        kde = gaussian_kde(values)
        x_grid = np.linspace(values.min(), values.max(), 400)
        ax.plot(x_grid, kde(x_grid), linewidth=2)

    ax.set_title("Daily Footfall Distribution")
    ax.set_xlabel("Daily Visitors")
    ax.set_ylabel("Density")
    save_figure(fig, "hist_footfall_distribution.png")


def chart_odor_vs_cleanliness(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.scatterplot(
        data=df,
        x="odor_score",
        y="cleanliness_score",
        hue="waste_level",
        style="waste_level",
        s=75,
        alpha=0.8,
        ax=ax,
    )

    x = df["odor_score"].to_numpy(dtype=float)
    y = df["cleanliness_score"].to_numpy(dtype=float)

    if len(x) >= 2 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            linewidth=2,
            linestyle="--",
            label="Linear trend",
        )

    ax.set_title("Odor Score vs. Cleanliness Score")
    ax.set_xlabel("Odor Score")
    ax.set_ylabel("Cleanliness Score")
    ax.legend()
    save_figure(fig, "scatter_odor_vs_cleanliness.png")


def chart_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_columns = [
        "cleanliness_score",
        "odor_score",
        "footfall",
        "complaints",
    ]

    correlation = df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Pearson Correlation Matrix")
    save_figure(fig, "heatmap_correlation_matrix.png")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Clean dataset not found. Run "
            "'python data-cleaning/clean_data.py' first."
        )

    df = load_data()

    chart_cleanliness_by_location(df)
    chart_complaints_by_water(df)
    chart_footfall_distribution(df)
    chart_odor_vs_cleanliness(df)
    chart_correlation_heatmap(df)

    print(f"Generated 5 charts in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
