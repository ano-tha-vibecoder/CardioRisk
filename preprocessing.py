"""The single shared raw-to-model feature transformation.

Used by both ``train.py`` and the web app so training and inference can never
drift apart. Categorical columns are encoded against their *full* training
domain (not whatever values happen to appear in the batch), which makes
single-row inference produce exactly the same columns as training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC_COLUMNS = ["age", "sex", "trestbps", "chol", "fbs", "thalach", "exang", "oldpeak", "ca"]

CATEGORIES = {
    "cp": [0, 1, 2, 3],
    "restecg": [0, 1, 2],
    "thal": [0, 1, 2],
    "slope": [0, 1, 2],
}

RAW_COLUMNS = NUMERIC_COLUMNS + list(CATEGORIES)


class ClevelandPreprocessor(BaseEstimator, TransformerMixin):
    """Domain-aware Cleveland encoding with a fixed output column order."""

    def fit(self, X, y=None):
        self._check_columns(X)
        self.feature_names_out_ = self._encode(X.iloc[:1]).columns.to_list()
        return self

    def transform(self, X):
        self._check_columns(X)
        frame = self._encode(X)
        if hasattr(self, "feature_names_out_"):
            frame = frame[self.feature_names_out_]
        return frame

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

    @staticmethod
    def _check_columns(X):
        missing = [c for c in RAW_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"missing input columns: {missing}")
        for column, categories in CATEGORIES.items():
            unknown = set(X[column].unique()) - set(categories)
            if unknown:
                # pd.Categorical would silently map these to NaN, i.e. an
                # all-zero dummy row that looks like the reference category.
                raise ValueError(f"unknown {column} values: {sorted(unknown)}")
        if (X["trestbps"] <= 0).any():
            raise ValueError("trestbps must be positive")

    @staticmethod
    def _encode(X):
        frame = X[RAW_COLUMNS].copy()
        frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].astype(float)
        for column, categories in CATEGORIES.items():
            frame[column] = pd.Categorical(frame[column], categories=categories)
        frame = pd.get_dummies(frame, columns=list(CATEGORIES), drop_first=True, dtype=float)
        frame["hr_bp_ratio"] = frame["thalach"] / frame["trestbps"]
        return frame
