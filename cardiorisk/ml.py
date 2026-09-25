"""Model loading, prediction and explanation for the CAD model."""
from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from preprocessing import CATEGORIES, RAW_COLUMNS
from preprocessing import ClevelandPreprocessor  # noqa: F401  (must be importable to unpickle)

from .config import ROOT

MODELS_DIR = ROOT / "models"


def load_model():
    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text())
    artifact = MODELS_DIR / metadata["artifact"]["file"]
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    # joblib/pickle executes code on load: refuse any file that does not match
    # the hash recorded at training time.
    if digest != metadata["artifact"]["sha256"]:
        raise RuntimeError(f"model artifact hash mismatch for {artifact.name}")
    return joblib.load(artifact), metadata


PIPELINE, MODEL_METADATA = load_model()
FEATURES = list(MODEL_METADATA["features"])

# Display metadata per *clinical* variable. One-hot columns (cp_1, cp_2, ...)
# are regrouped under their source variable: SHAP values of a linear model are
# additive, so the group sum is the exact contribution of that variable.
VARIABLES = {
    "age": ("Age", "{:g} yrs"),
    "sex": ("Sex", {0: "Female", 1: "Male"}),
    "trestbps": ("Resting blood pressure", "{:g} mmHg"),
    "chol": ("Serum cholesterol", "{:g} mg/dl"),
    "fbs": ("Fasting blood sugar > 120", {0: "No", 1: "Yes"}),
    "thalach": ("Max heart rate", "{:g} bpm"),
    "exang": ("Exercise angina", {0: "No", 1: "Yes"}),
    "oldpeak": ("ST depression", "{:g} mm"),
    "ca": ("Major vessels coloured", "{:g}"),
    "cp": ("Chest pain type", {0: "Typical angina", 1: "Atypical angina",
                               2: "Non-anginal pain", 3: "Asymptomatic"}),
    "restecg": ("Resting ECG", {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}),
    "slope": ("ST slope", {0: "Upsloping", 1: "Flat", 2: "Downsloping"}),
    "thal": ("Thallium test", {0: "Normal", 1: "Fixed defect", 2: "Reversible defect"}),
    "hr_bp_ratio": ("Heart rate / BP ratio", "{:.2f}"),
}


def source_variable(feature: str) -> str:
    base = feature.rsplit("_", 1)[0]
    return base if base in CATEGORIES else feature


if unknown := {source_variable(f) for f in FEATURES} - set(VARIABLES):
    raise RuntimeError(f"no display metadata for model features: {sorted(unknown)}")


def explain(features) -> dict[str, float]:
    """Per clinical variable contribution to the log-odds vs the cohort mean.

    These are exact interventional SHAP values for a linear model against the
    training background, coef * (x - E[x]); the scaler was fit on that same
    background, so E[x] is 0 in scaled space.
    """
    per_feature = PIPELINE[-1].coef_[0] * np.asarray(features)[0]
    grouped: dict[str, float] = {}
    for name, value in zip(FEATURES, per_feature):
        key = source_variable(name)
        grouped[key] = grouped.get(key, 0.0) + float(value)
    return grouped


def predict(values: dict) -> dict:
    raw = pd.DataFrame([values], columns=RAW_COLUMNS)
    features = PIPELINE[:-1].transform(raw)
    model = PIPELINE[-1]
    prob = float(model.predict_proba(features)[0, 1])

    top = sorted(explain(features).items(), key=lambda p: abs(p[1]), reverse=True)[:6]
    scale = max(abs(v) for _, v in top) or 1.0
    shown = {**values, "hr_bp_ratio": values["thalach"] / values["trestbps"]}

    bands = MODEL_METADATA["risk_bands"]
    risk = "High" if prob >= bands["high"] else "Moderate" if prob >= bands["moderate"] else "Low"
    return {
        "model_version": MODEL_METADATA["model_version"],
        "probability": prob,
        "positive": prob >= MODEL_METADATA["decision_threshold"],
        "risk": risk,
        "contributions": [
            {"label": VARIABLES[name][0], "shown": _format(name, shown[name]), "value": v,
             "width": max(abs(v) / scale * 100, 4.0)}
            for name, v in top
        ],
    }


def _format(variable: str, value) -> str:
    fmt = VARIABLES[variable][1]
    return fmt[int(value)] if isinstance(fmt, dict) else fmt.format(value)
