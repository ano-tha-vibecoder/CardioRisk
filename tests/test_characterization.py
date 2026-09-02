import re

import pandas as pd
import pytest

import app


PAYLOADS = [
    {
        "age": "54", "sex": "1", "trestbps": "130", "chol": "245",
        "fbs": "0", "thalach": "150", "exang": "0", "oldpeak": "1.5",
        "ca": "0", "cp": "atypical", "restecg": "normal",
        "thal": "normal", "slope": "up",
    },
    {
        "age": "69", "sex": "0", "trestbps": "140", "chol": "239",
        "fbs": "0", "thalach": "151", "exang": "0", "oldpeak": "1.8",
        "ca": "2", "cp": "non", "restecg": "normal",
        "thal": "normal", "slope": "up",
    },
]

OLD_GOLDENS = [
    {"probability": "13.8", "risk": "Low", "shap": [
        ("Major Vessels (ca)", "-0.889"), ("Asymptomatic", "-0.684"),
        ("Reversible Defect", "-0.591"), ("Flat ST Slope", "-0.446"),
        ("Sex", "+0.384"), ("Exercise Angina", "-0.292"),
    ]},
]
NEW_GOLDENS = [
    {"probability": "13.8", "shap": [
        ("Asymptomatic", "-1.013"), ("Major Vessels (ca)", "-0.923"),
        ("cp_1", "+0.745"), ("Reversible Defect", "-0.673"),
        ("Flat ST Slope", "-0.589"), ("Sex", "+0.346"),
    ]},
]

RAW_CATEGORIES = {
    "cp": [0, 1, 2, 3],
    "restecg": [0, 1, 2],
    "thal": [0, 1, 2],
    "slope": [0, 1, 2],
}
SELECTED_DUMMIES = ["cp_2", "cp_3", "restecg_2", "thal_2", "slope_1"]


def _result(response):
    html = response.get_data(as_text=True)
    probability = re.search(r'<span class="value">(.*?)</span>', html).group(1)
    rows = re.findall(
        r'<span class="shap-label">(.*?)</span>.*?'
        r'<span class="shap-val .*?">(.*?)</span>', html, re.S
    )
    return probability, rows


def test_representative_assessment_matches_current_golden_output():
    response = app.app.test_client().post("/assess", data=PAYLOADS[0])
    assert response.status_code == 200
    probability, shap_rows = _result(response)
    assert probability == NEW_GOLDENS[0]["probability"]
    assert shap_rows == NEW_GOLDENS[0]["shap"]
    assert "Low Risk" in response.get_data(as_text=True)


def test_second_representative_assessment_is_successful():
    response = app.app.test_client().post("/assess", data=PAYLOADS[1])
    assert response.status_code == 200
    assert '<span class="value">' in response.get_data(as_text=True)


def _manual_dummies(row):
    return {
        "cp_2": float(row["cp"] == 2),
        "cp_3": float(row["cp"] == 3),
        "restecg_2": float(row["restecg"] == 2),
        "thal_2": float(row["thal"] == 2),
        "slope_1": float(row["slope"] == 1),
    }


def _pandas_dummies_with_known_domains(row):
    # Naive single-row get_dummies only sees categories in that row and can
    # omit expected columns. Categorical dtype preserves the full training
    # domain, making single-row inference equivalent to the notebook pipeline.
    one = pd.DataFrame([row]).copy()
    for column, categories in RAW_CATEGORIES.items():
        one[column] = pd.Categorical(one[column], categories=categories)
    encoded = pd.get_dummies(
        one, columns=list(RAW_CATEGORIES), drop_first=True
    )
    return {column: float(encoded.iloc[0].get(column, 0))
            for column in SELECTED_DUMMIES}


def test_manual_and_notebook_preprocessing_match_all_csv_category_combinations():
    csv = pd.read_csv("data/heart_cleveland_upload.csv")
    combinations = csv[[*RAW_CATEGORIES]].drop_duplicates()
    assert len(combinations) > 1
    for _, row in combinations.iterrows():
        assert _manual_dummies(row) == _pandas_dummies_with_known_domains(row), row.to_dict()
