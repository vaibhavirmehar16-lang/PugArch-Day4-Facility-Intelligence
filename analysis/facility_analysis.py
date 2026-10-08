"""
Facility operational analysis using Pandas and NumPy.

Usage:
    python analysis/facility_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "dataset" / "cleaned_facility_data.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "analysis_report.json"


def load_data() -> pd.DataFrame:
    """Load and prepare cleaned facility data."""
    df = pd.read_csv(CLEAN_PATH, parse_dates=["inspection_date"])
    return df


def run_analysis(df: pd.DataFrame) -> dict:
    """Run the required operational analysis and return serializable results."""
    location_summary = (
        df.groupby("location")
        .agg(
            mean_cleanliness=("cleanliness_score", "mean"),
            mean_odor=("odor_score", "mean"),
            mean_footfall=("footfall", "mean"),
            total_complaints=("complaints", "sum"),
            facility_count=("facility_id", "count"),
        )
        .sort_values("mean_cleanliness")
    )

    water_summary = (
        df.groupby("water_availability")
        .agg(
            total_complaints=("complaints", "sum"),
            average_complaints=("complaints", "mean"),
            average_cleanliness=("cleanliness_score", "mean"),
            average_odor=("odor_score", "mean"),
            facility_count=("facility_id", "count"),
        )
        .sort_index()
    )

    waste_summary = (
        df.groupby("waste_level")
        .agg(
            mean_footfall=("footfall", "mean"),
            mean_complaints=("complaints", "mean"),
            mean_cleanliness=("cleanliness_score", "mean"),
            facility_count=("facility_id", "count"),
        )
        .reindex(["Low", "Medium", "High", "Critical"])
    )

    numeric_columns = [
        "cleanliness_score",
        "odor_score",
        "footfall",
        "complaints",
    ]
    correlation_matrix = df[numeric_columns].corr(method="pearson")

    odor_complaints_corr = float(
        df["odor_score"].corr(df["complaints"])
    )

    # Water impact: compare available vs unavailable.
    if {"Yes", "No"}.issubset(set(water_summary.index)):
        yes = water_summary.loc["Yes"]
        no = water_summary.loc["No"]

        cleanliness_difference = float(
            yes["average_cleanliness"] - no["average_cleanliness"]
        )
        complaint_difference = float(
            no["average_complaints"] - yes["average_complaints"]
        )
    else:
        cleanliness_difference = 0.0
        complaint_difference = 0.0

    # Risk score balances high footfall and poor cleanliness.
    footfall_min = df["footfall"].min()
    footfall_max = df["footfall"].max()
    clean_min = df["cleanliness_score"].min()
    clean_max = df["cleanliness_score"].max()

    footfall_range = max(footfall_max - footfall_min, 1)
    clean_range = max(clean_max - clean_min, 1)

    risk_df = (
        df.groupby("location")
        .agg(
            avg_footfall=("footfall", "mean"),
            avg_cleanliness=("cleanliness_score", "mean"),
            avg_complaints=("complaints", "mean"),
        )
        .copy()
    )

    risk_df["footfall_component"] = (
        (risk_df["avg_footfall"] - footfall_min) / footfall_range
    )
    risk_df["cleanliness_risk_component"] = (
        (clean_max - risk_df["avg_cleanliness"]) / clean_range
    )
    risk_df["risk_score"] = (
        0.60 * risk_df["footfall_component"]
        + 0.40 * risk_df["cleanliness_risk_component"]
    )

    highest_risk_location = risk_df["risk_score"].idxmax()
    highest_risk = risk_df.loc[highest_risk_location]

    insights = [
        {
            "title": "Odor and Complaints",
            "metric": odor_complaints_corr,
            "text": (
                f"The Pearson correlation between odor score and complaints "
                f"is {odor_complaints_corr:.2f}. "
                "A positive value indicates that higher odor severity tends "
                "to be associated with more registered complaints."
            ),
        },
        {
            "title": "Water Availability Impact",
            "metric": cleanliness_difference,
            "text": (
                f"Facilities with water availability show a "
                f"{cleanliness_difference:.2f}-point cleanliness difference "
                f"versus facilities without water availability. "
                f"The complaint-rate gap is {complaint_difference:.2f} "
                "complaints per facility in favor of water-available sites."
            ),
        },
        {
            "title": "Highest-Risk Location",
            "metric": float(highest_risk["risk_score"]),
            "text": (
                f"{highest_risk_location} is the highest-risk location under "
                f"the weighted operational risk score, with average footfall "
                f"of {highest_risk['avg_footfall']:.0f}, average cleanliness "
                f"of {highest_risk['avg_cleanliness']:.2f}, and average "
                f"complaints of {highest_risk['avg_complaints']:.2f}."
            ),
        },
    ]

    return {
        "dataset_summary": {
            "rows": int(len(df)),
            "facilities": int(df["facility_id"].nunique()),
            "average_cleanliness": float(df["cleanliness_score"].mean()),
            "average_odor": float(df["odor_score"].mean()),
            "total_complaints": int(df["complaints"].sum()),
            "water_availability_rate": float(
                (df["water_availability"] == "Yes").mean() * 100
            ),
        },
        "location_summary": location_summary.round(3).to_dict("index"),
        "water_summary": water_summary.round(3).to_dict("index"),
        "waste_summary": waste_summary.round(3).to_dict("index"),
        "correlation_matrix": correlation_matrix.round(4).to_dict(),
        "odor_complaints_correlation": odor_complaints_corr,
        "risk_by_location": risk_df.round(3).to_dict("index"),
        "insights": insights,
    }


def print_report(report: dict) -> None:
    """Print a readable analytical report."""
    print("\n" + "=" * 75)
    print("FACILITY OPERATIONAL INTELLIGENCE REPORT")
    print("=" * 75)

    summary = report["dataset_summary"]
    print(f"\nFacilities: {summary['facilities']}")
    print(f"Average cleanliness: {summary['average_cleanliness']:.2f}")
    print(f"Average odor score: {summary['average_odor']:.2f}")
    print(f"Total complaints: {summary['total_complaints']}")
    print(f"Water availability: {summary['water_availability_rate']:.2f}%")

    print("\n--- Average Quality by Location ---")
    print(pd.DataFrame(report["location_summary"]).T)

    print("\n--- Complaints & Hygiene by Water Availability ---")
    print(pd.DataFrame(report["water_summary"]).T)

    print("\n--- Waste Level Summary ---")
    print(pd.DataFrame(report["waste_summary"]).T)

    print("\n--- Pearson Correlation Matrix ---")
    print(pd.DataFrame(report["correlation_matrix"]))

    print("\n--- Operational Insights ---")
    for index, insight in enumerate(report["insights"], start=1):
        print(f"{index}. {insight['text']}")


def main() -> None:
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            "Clean dataset not found. Run "
            "'python data-cleaning/clean_data.py' first."
        )

    df = load_data()
    report = run_analysis(df)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print_report(report)
    print(f"\nAnalysis report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
