from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "cleaned_facility_data.csv"
OUT = ROOT / "evaluation" / "facility_segments.csv"
MODEL = ROOT / "models" / "facility_kmeans.joblib"
FEATURES = ["cleanliness_score", "odor_score", "footfall", "complaints", "hours_since_cleaning"]

def main():
    df = pd.read_csv(DATA)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    model = KMeans(n_clusters=3, random_state=42, n_init=20)
    df["segment"] = model.fit_predict(X)
    centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=FEATURES).round(2)
    centers["segment"] = centers.index
    centers.to_csv(ROOT / "evaluation" / "segment_profiles.csv", index=False)
    df[["facility_id", "location", "segment"]].to_csv(OUT, index=False)
    joblib.dump({"model": model, "scaler": scaler, "features": FEATURES}, MODEL)
    print("K-Means segmentation complete.")

if __name__ == "__main__":
    main()
