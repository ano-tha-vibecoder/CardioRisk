"""Deterministically train, evaluate and serialize the CardioRisk model.

Evaluation uses repeated stratified k-fold cross-validation of the *whole*
pipeline (preprocessing, scaling and model are refit in every fold), so the
reported metrics are not inflated by leakage or by one lucky split. The
production model is then refit on all available data.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import RAW_COLUMNS, ClevelandPreprocessor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/heart_cleveland_upload.csv"
MODELS = ROOT / "models"
ARTIFACT = MODELS / "cardio_pipeline.joblib"
METADATA = MODELS / "model_metadata.json"
SEED = 42
VERSION = "3.0.0"
# L2 strength chosen between C in {0.1, 1}: equal AUC, slightly better Brier.
C = 0.1
CV_SPLITS, CV_REPEATS = 5, 10
DECISION_THRESHOLD = 0.5
RISK_BANDS = {"moderate": 0.35, "high": 0.65}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocess", ClevelandPreprocessor()),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=1000, random_state=SEED)),
    ])


def evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    scoring = {
        "auc": "roc_auc",
        "brier": "neg_brier_score",
        "accuracy": "accuracy",
        "f1": "f1",
        "sensitivity": "recall",
        "specificity": make_scorer(recall_score, pos_label=0),
    }
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED)
    scores = cross_validate(build_pipeline(), X, y, cv=cv, scoring=scoring)
    metrics = {}
    for name in scoring:
        values = scores[f"test_{name}"]
        if name == "brier":
            values = -values
        metrics[name] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return metrics


def main():
    MODELS.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA)
    X, y = raw[RAW_COLUMNS], raw["condition"]

    metrics = evaluate(X, y)
    pipeline = build_pipeline().fit(X, y)
    joblib.dump(pipeline, ARTIFACT)

    metadata = {
        "model_version": VERSION,
        "model_type": "L2-regularised logistic regression",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_data": {
            "source": "UCI Cleveland heart disease (297 complete cases)",
            "sha256": sha256(DATA),
            "n_samples": int(len(y)),
            "prevalence": float(y.mean()),
            "target": "angiographic coronary artery disease (>50% diameter narrowing)",
        },
        "artifact": {"file": ARTIFACT.name, "sha256": sha256(ARTIFACT),
                     "sklearn_version": sklearn.__version__},
        "evaluation": {
            "method": f"repeated stratified {CV_SPLITS}-fold CV x {CV_REPEATS} repeats",
            "decision_threshold": DECISION_THRESHOLD,
            "metrics": metrics,
        },
        "decision_threshold": DECISION_THRESHOLD,
        "risk_bands": RISK_BANDS,
        "features": pipeline.named_steps["preprocess"].feature_names_out_,
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
