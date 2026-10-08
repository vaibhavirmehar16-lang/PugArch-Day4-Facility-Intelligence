"""
Dataset Generator: Facility Monitoring Operations (Indian Transit Hubs)
Generates realistic facility data with controlled anomalies, missing values,
and statistical outliers for the Day 4 ML classification pipeline.
"""

from datetime import date, timedelta
import os
import random
import numpy as np
import pandas as pd

SEED = 42

LOCATIONS = [
    'CSMT Concourse',
    'IGI T3 Terminal',
    'Howrah Main Hub',
    'Electronic City Hub',
    'Connaught Place Metro'
]

WASTE_LEVELS = ["Low", "Medium", "High", "Critical"]
WATER_CHOICES = ["Yes", "No"]


def generate_dataset() -> pd.DataFrame:
    """Generate clean base records, then inject controlled dirty data."""
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    n_samples = 150
    today = date.today()

    records = []
    for i in range(1, n_samples + 1):
        facility_id = f"FAC_{i:03d}"
        location = random.choice(LOCATIONS)
        days_ago = random.randint(0, 90)
        inspection_date = today - timedelta(days=days_ago)

        water_availability = "Yes" if random.random() < 0.82 else "No"

        # Nominal footfall between 300 and 2600
        footfall = int(rng.normal(loc=1400, scale=450))
        footfall = max(250, min(footfall, 2800))

        # Base cleanliness (1.0 to 10.0)
        base_clean = rng.normal(loc=7.2, scale=1.3)
        if water_availability == "No":
            base_clean -= 2.2
        if footfall > 1900:
            base_clean -= 0.9
        cleanliness_score = round(float(np.clip(base_clean, 1.0, 10.0)), 2)

        # Odor score (1.0 to 5.0) inversely related to cleanliness
        base_odor = 6.2 - (cleanliness_score * 0.52) + rng.normal(0, 0.35)
        odor_score = round(float(np.clip(base_odor, 1.0, 5.0)), 2)

        # Hours since last cleaning (0 to 72 hours)
        hours_since = int(rng.exponential(scale=18))
        hours_since = max(1, min(hours_since, 72))

        # Waste level correlated with cleanliness and footfall
        if cleanliness_score > 8.0 and footfall < 1200:
            waste_level = random.choices(["Low", "Medium"], weights=[0.8, 0.2])[0]
        elif cleanliness_score < 5.0 or footfall > 2000 or hours_since > 36:
            waste_level = random.choices(["High", "Critical"], weights=[0.55, 0.45])[0]
        else:
            waste_level = random.choices(WASTE_LEVELS, weights=[0.25, 0.45, 0.20, 0.10])[0]

        # Complaints logged
        base_comp = (
            (10.0 - cleanliness_score) * 1.6 +
            (odor_score * 2.1) +
            (hours_since * 0.15) +
            (6.5 if water_availability == "No" else 0.0) +
            rng.normal(0, 1.8)
        )
        complaints = int(np.clip(round(base_comp), 0, 30))

        records.append({
            "facility_id": facility_id,
            "location": location,
            "cleanliness_score": cleanliness_score,
            "odor_score": odor_score,
            "waste_level": waste_level,
            "water_availability": water_availability,
            "footfall": footfall,
            "complaints": complaints,
            "hours_since_cleaning": hours_since,
            "inspection_date": inspection_date.isoformat()
        })

    df = pd.DataFrame(records)

    # Synthetic ML ground truth target: High vs Low hygiene risk
    # Target rule: low cleanliness (<5.5), high odor (>3.2), critical waste, or excessive complaints
    is_high_risk = (
        (df["cleanliness_score"] < 5.5) |
        (df["odor_score"] > 3.2) |
        (df["waste_level"].isin(["High", "Critical"])) |
        (df["complaints"] >= 10) |
        ((df["water_availability"] == "No") & (df["hours_since_cleaning"] > 20))
    )
    df["hygiene_risk"] = np.where(is_high_risk, "High", "Low")

    # --- Controlled Anomaly & Dirty Data Injections ---
    # 1. Inject duplicate rows (5 duplicates)
    dups = df.sample(n=5, random_state=SEED)
    df = pd.concat([df, dups], ignore_index=True)

    # 2. Inject missing values (~8% in key fields)
    nan_idx = rng.choice(df.index, size=int(len(df) * 0.08), replace=False)
    df.loc[nan_idx, "cleanliness_score"] = np.nan

    nan_odor = rng.choice(df.index, size=int(len(df) * 0.08), replace=False)
    df.loc[nan_odor, "odor_score"] = np.nan

    nan_water = rng.choice(df.index, size=int(len(df) * 0.08), replace=False)
    df.loc[nan_water, "water_availability"] = np.nan

    # 3. Inject negative complaints (invalid bounds)
    neg_idx = rng.choice(df.index, size=3, replace=False)
    df.loc[neg_idx, "complaints"] = [-3, -1, -5]

    # 4. Inject extreme statistical footfall outliers
    outlier_idx = rng.choice(df.index, size=4, replace=False)
    df.loc[outlier_idx[0], "footfall"] = 19200
    df.loc[outlier_idx[1], "footfall"] = 24500
    df.loc[outlier_idx[2], "footfall"] = -450
    df.loc[outlier_idx[3], "footfall"] = 17800

    return df


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(base_dir, "raw_facility_data.csv")
    df = generate_dataset()
    df.to_csv(out_file, index=False)
    print(f"[OK] Raw facility dataset generated: {out_file} (Shape: {df.shape})")


if __name__ == "__main__":
    main()