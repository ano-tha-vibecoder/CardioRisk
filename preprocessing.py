"""The single shared raw-to-model feature transformation."""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CATEGORIES = {
    "cp": [0, 1, 2, 3],
    "restecg": [0, 1, 2],
    "thal": [0, 1, 2],
    "slope": [0, 1, 2],
}

class ClevelandPreprocessor(BaseEstimator, TransformerMixin):
    """Domain-aware Cleveland encoding, preserving training columns."""
    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        frame = X.copy()
        for column, categories in CATEGORIES.items():
            frame[column] = pd.Categorical(frame[column], categories=categories)
        frame = pd.get_dummies(frame, columns=list(CATEGORIES), drop_first=True)
        frame["hr_bp_ratio"] = frame["thalach"] / frame["trestbps"]
        return frame

    def get_feature_names_out(self, input_features=None):
        return self.transform(pd.DataFrame([{c: CATEGORIES[c][0] if c in CATEGORIES else 0 for c in self.feature_names_in_}])).columns.to_numpy()
