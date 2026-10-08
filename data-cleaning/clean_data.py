"""
Automated data-quality and cleaning pipeline for the facility dataset.

Usage:
    python data-cleaning/clean_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "dataset" / "raw_facility_data.csv"
CLEAN_PATH = PROJECT_ROOT / "dataset" / "cleaned_facility_data.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "data_quality_report.json"


def inspect_data(df: pd.DataFrame) -> None:
    """Print schema and null counts."""
    print("\n" + "=" * 70)
    print("RAW DATA INSPECTION")
    print("=" * 70)
    print("\nData types:\n")
    print(df.dtypes)
    print("\nNull value counts:\n")
    print(df.isna().sum())


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the dataset according to the assignment specification."""
    before_rows = len(df)

    # Ensure numeric columns are numeric.
    numeric_columns = [
        "cleanliness_score",
        "odor_score",
        "footfall",
        "complaints",
        "hours_since_cleaning",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Standardize categorical values.
    df["location"] = df["location"].astype("string").str.strip()
    df["waste_level"] = df["waste_level"].astype("string").str.strip()
    df["water_availability"] = (
        df["water_availability"].astype("string").str.strip()
    )

    # Remove exact duplicate records.
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Replace invalid negative complaint counts with zero.
    invalid_complaints = int((df["complaints"] < 0).sum())
    df.loc[df["complaints"] < 0, "complaints"] = 0

    # Group-wise median imputation by location.
    for column in ["cleanliness_score", "odor_score"]:
        global_median = df[column].median()
        df[column] = df.groupby("location", dropna=False)[column].transform(
            lambda s: s.fillna(s.median())
        )
        df[column] = df[column].fillna(global_median)

    # Categorical mode imputation.
    water_mode = df["water_availability"].mode(dropna=True)
    water_fill_value = water_mode.iloc[0] if not water_mode.empty else "Yes"
    missing_water_before = int(df["water_availability"].isna().sum())
    df["water_availability"] = df["water_availability"].fillna(
        water_fill_value
    )

    # Footfall outlier capping using NumPy IQR.
    footfall_values = df["footfall"].dropna().to_numpy(dtype=float)
    q1 = float(np.percentile(footfall_values, 25))
    q3 = float(np.percentile(footfall_values, 75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (df["footfall"] < lower_bound)
        | (df["footfall"] > upper_bound)
    )
    footfall_outliers_capped = int(outlier_mask.sum())

    df["footfall"] = np.clip(
        df["footfall"],
        lower_bound,
        upper_bound,
    ).round().astype(int)

    # Standardize cleaning-time feature.
    df["hours_since_cleaning"] = pd.to_numeric(
        df["hours_since_cleaning"], errors="coerce"
    ).clip(lower=0, upper=168)
    hours_median = df["hours_since_cleaning"].median()
    df["hours_since_cleaning"] = df["hours_since_cleaning"].fillna(hours_median).round().astype(int)

    # Standardize inspection date.
    df["inspection_date"] = pd.to_datetime(
        df["inspection_date"],
        errors="coerce",
    )

    # Final type normalization.
    df["facility_id"] = df["facility_id"].astype(str)
    df["complaints"] = df["complaints"].round().astype(int)
    df["cleanliness_score"] = df["cleanliness_score"].round(2)
    df["odor_score"] = df["odor_score"].round(2)

    # Synthetic educational target for Day 4 ML. In a production system this
    # should be replaced with historical labeled outcomes (e.g. failed inspection).
    rng = np.random.default_rng(42)
    clean_risk = (10 - df["cleanliness_score"]) / 9
    odor_risk = (df["odor_score"] - 1) / 4
    waste_risk = df["waste_level"].map({"Low": 0.0, "Medium": 0.35, "High": 0.70, "Critical": 1.0})
    water_risk = df["water_availability"].map({"Yes": 0.0, "No": 0.35})
    footfall_risk = np.clip((df["footfall"] - 200) / 2300, 0, 1)
    complaint_risk = np.clip(df["complaints"] / 30, 0, 1)
    cleaning_delay_risk = np.clip((df["hours_since_cleaning"] - 2) / 94, 0, 1)
    latent_risk = (
        1.5 * clean_risk + 1.1 * odor_risk + 1.4 * waste_risk
        + 0.5 * water_risk + 0.7 * footfall_risk
        + 0.9 * complaint_risk + 1.0 * cleaning_delay_risk
    )
    probability = 1 / (1 + np.exp(-(latent_risk - 2.4) * 2.0))
    df["hygiene_risk"] = np.where(rng.random(len(df)) < probability, "High", "Low")

    after_rows = len(df)

    report = {
        "rows_before_cleaning": before_rows,
        "rows_after_cleaning": after_rows,
        "duplicate_rows_removed": duplicate_count,
        "negative_complaints_replaced_with_zero": invalid_complaints,
        "missing_water_values_imputed": missing_water_before,
        "footfall_q1": round(q1, 2),
        "footfall_q3": round(q3, 2),
        "footfall_iqr": round(iqr, 2),
        "footfall_lower_bound": round(lower_bound, 2),
        "footfall_upper_bound": round(upper_bound, 2),
        "footfall_outliers_capped": footfall_outliers_capped,
        "remaining_missing_values": int(df.isna().sum().sum()),
        "ml_target": "hygiene_risk",
        "ml_target_note": "Synthetic educational target generated after cleaning; replace with historical labeled outcomes for production use.",
    }

    return df, report


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_PATH}\n"
            "Run: python dataset/generate_dataset.py"
        )

    df = pd.read_csv(RAW_PATH)
    inspect_data(df)

    cleaned_df, report = clean_data(df)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(CLEAN_PATH, index=False)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("CLEANING REPORT")
    print("=" * 70)

    for key, value in report.items():
        print(f"{key}: {value}")

    print(f"\nClean dataset saved to: {CLEAN_PATH}")
    print(f"Quality report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
