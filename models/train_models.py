from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "cleaned_facility_data.csv"
MODEL_DIR = ROOT / "models"
EVAL_DIR = ROOT / "evaluation"
PRED_DIR = ROOT / "predictions"

FEATURES = ["cleanliness_score", "odor_score", "waste_level", "water_availability", "footfall", "complaints", "hours_since_cleaning"]
TARGET = "hygiene_risk"
NUMERIC = ["cleanliness_score", "odor_score", "footfall", "complaints", "hours_since_cleaning"]
CATEGORICAL = ["waste_level", "water_availability"]


def build_pipeline(model):
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("preprocessor", preprocess), ("model", model)])


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, list(model.classes_).index("High")]
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=["Low", "High"]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, pos_label="High", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, pos_label="High", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, pred, pos_label="High", zero_division=0)), 4),
        "mae_probability": round(float(mean_absolute_error((y_test == "High").astype(int), prob)), 4),
        "mse_probability": round(float(mean_squared_error((y_test == "High").astype(int), prob)), 4),
        "rmse_probability": round(float(mean_squared_error((y_test == "High").astype(int), prob) ** 0.5), 4),
        "confusion_matrix": {"labels": ["Low", "High"], "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]},
    }


def main():
    df = pd.read_csv(DATA)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        "Logistic Regression": build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": build_pipeline(RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", min_samples_leaf=2)),
    }

    results = {}
    best_name, best_model, best_f1 = None, None, -1
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        if metrics["f1_score"] > best_f1:
            best_name, best_model, best_f1 = name, model, metrics["f1_score"]

    MODEL_DIR.mkdir(exist_ok=True)
    EVAL_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "model_name": best_name, "features": FEATURES}, MODEL_DIR / "best_hygiene_risk_model.joblib")

    test_out = X_test.copy()
    test_out["actual_risk"] = y_test.values
    test_out["predicted_risk"] = best_model.predict(X_test)
    test_out["high_risk_probability"] = best_model.predict_proba(X_test)[:, list(best_model.classes_).index("High")]
    test_out.to_csv(PRED_DIR / "test_predictions.csv", index=False)

    # Confusion matrix image for internship documentation.
    cm = results[best_name]["confusion_matrix"]["matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)
    ax.set_xticks([0, 1], ["Low", "High"])
    ax.set_yticks([0, 1], ["Low", "High"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "target": TARGET,
        "features": FEATURES,
        "test_size": 0.20,
        "random_state": 42,
        "best_model": best_name,
        "models": results,
        "note": "The hygiene_risk label is a synthetic educational target generated from operational signals; it demonstrates the complete ML workflow and should be replaced by historical labeled outcomes in production.",
    }
    (EVAL_DIR / "model_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
