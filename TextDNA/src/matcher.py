"""Training and inference utilities for candidate-job matching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def _build_model(model_name: str, random_state: int = 42):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=3000, random_state=random_state)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    return GradientBoostingClassifier(random_state=random_state)


def train_match_model(feature_df: pd.DataFrame, labels: np.ndarray, config: dict):
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df.values,
        labels,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=labels,
    )

    model = _build_model(
        config["training"].get("model", "gradient_boosting"),
        random_state=config["training"]["random_state"],
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "avg_predicted_probability": float(np.mean(probs)),
    }

    return model, metrics


def save_model_artifacts(model, feature_columns, config: dict, metrics: Dict[str, float]) -> None:
    root = Path(__file__).parent.parent
    model_dir = root / config["paths"]["models"]
    results_dir = root / config["paths"]["results"]
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "feature_columns": list(feature_columns)}, model_dir / "match_model.joblib")

    with open(results_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_model_artifacts(config: dict):
    root = Path(__file__).parent.parent
    model_path = root / config["paths"]["models"] / "match_model.joblib"
    return joblib.load(model_path)
