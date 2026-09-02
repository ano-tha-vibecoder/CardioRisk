"""Deterministically train and serialize the CardioRisk model."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from preprocessing import ClevelandPreprocessor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/heart_cleveland_upload.csv"
MODELS = ROOT / "models"
SEED = 42
VERSION = "2.0.0"


def main():
    MODELS.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA)
    X_raw = raw.drop(columns=["condition"])
    y = raw["condition"]
    preprocessor = ClevelandPreprocessor().fit(X_raw)
    X = preprocessor.transform(X_raw)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    selector = RFECV(
        estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        step=1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="roc_auc", min_features_to_select=5,
    ).fit(x_train, y_train)
    selected = x_train.columns[selector.support_].tolist()

    scaler_eval = StandardScaler().fit(x_train[selected])
    smote = SMOTE(random_state=SEED)
    x_bal, y_bal = smote.fit_resample(scaler_eval.transform(x_train[selected]), y_train)
    eval_model = LogisticRegression(max_iter=1000, C=1, random_state=SEED).fit(x_bal, y_bal)
    test_scaled = scaler_eval.transform(x_test[selected])
    probabilities = eval_model.predict_proba(test_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "auc": roc_auc_score(y_test, probabilities),
        "f1": f1_score(y_test, predictions),
    }

    # Production artifacts are fit deterministically on all available data.
    scaler = StandardScaler().fit(X[selected])
    x_bal, y_bal = SMOTE(random_state=SEED).fit_resample(scaler.transform(X[selected]), y)
    model = LogisticRegression(max_iter=1000, C=1, random_state=SEED).fit(x_bal, y_bal)
    joblib.dump(model, MODELS / "best_model.pkl")
    joblib.dump(scaler, MODELS / "scaler.pkl")
    joblib.dump(selected, MODELS / "selected_features.pkl")
    joblib.dump(preprocessor, MODELS / "preprocessor.pkl")

    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    metadata = {
        "model_version": VERSION,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_data_sha256": digest,
        "evaluation": {k: float(v) for k, v in metrics.items()},
        "evaluation_split": {"test_size": 0.2, "random_state": SEED, "stratified": True},
        "selected_features": selected,
    }
    (MODELS / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
