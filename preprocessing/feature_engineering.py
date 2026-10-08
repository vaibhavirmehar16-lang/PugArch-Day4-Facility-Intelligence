from __future__ import annotations

import pandas as pd

FEATURES = [
    "cleanliness_score", "odor_score", "waste_level",
    "water_availability", "footfall", "complaints", "hours_since_cleaning"
]
TARGET = "hygiene_risk"


def validate_features(df: pd.DataFrame) -> None:
    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    validate_features(df)
    return df[FEATURES + [TARGET]].copy()
