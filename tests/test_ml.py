import numpy as np
import pandas as pd
import pytest

import train
from cardiorisk import ml
from cardiorisk.validation import validate
from preprocessing import CATEGORIES, RAW_COLUMNS, ClevelandPreprocessor
from tests.conftest import ROOT, VALID_RAW

DATA = pd.read_csv(ROOT / "data/heart_cleveland_upload.csv")


def test_representative_prediction_golden_output():
    values, errors, _ = validate(VALID_RAW)
    assert not errors
    result = ml.predict(values)
    assert round(result["probability"] * 100, 1) == 15.8
    assert result["risk"] == "Low" and not result["positive"]
    assert [(c["label"], c["shown"]) for c in result["contributions"][:3]] == [
        ("Major vessels coloured", "0"), ("Thallium test", "Normal"), ("ST slope", "Upsloping")]


def test_shipped_artifact_matches_training_code():
    """Guards against a stale artifact that train.py would no longer produce."""
    X, y = DATA[RAW_COLUMNS], DATA["condition"]
    fresh = train.build_pipeline().fit(X, y)
    np.testing.assert_allclose(fresh.predict_proba(X), ml.PIPELINE.predict_proba(X))


def test_contributions_match_shap_linear_explainer():
    shap = pytest.importorskip("shap")
    pipeline = ml.PIPELINE
    background = pipeline[:-1].transform(DATA[RAW_COLUMNS])
    # shap's default masker subsamples 100 background rows; use them all.
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.LinearExplainer(pipeline[-1], masker)
    row = DATA[RAW_COLUMNS].iloc[[5]]
    expected = explainer.shap_values(pipeline[:-1].transform(row))[0]
    manual = pipeline[-1].coef_[0] * pipeline[:-1].transform(row)[0]
    np.testing.assert_allclose(manual, expected, atol=1e-9)


def test_grouped_contributions_are_exact():
    values, _, _ = validate(VALID_RAW)
    features = ml.PIPELINE[:-1].transform(pd.DataFrame([values], columns=RAW_COLUMNS))
    grouped = ml.explain(features)
    assert set(grouped) == {*RAW_COLUMNS, "hr_bp_ratio"}
    # SHAP efficiency: contributions sum to f(x) - E[f(x)] in log-odds.
    assert sum(grouped.values()) == pytest.approx(float(ml.PIPELINE[-1].coef_[0] @ features[0]))


def test_prediction_is_json_serialisable():
    import json
    values, _, _ = validate(VALID_RAW)
    json.dumps(ml.predict(values))


def test_single_row_encoding_matches_batch_encoding():
    pre = ClevelandPreprocessor().fit(DATA[RAW_COLUMNS])
    batch = pre.transform(DATA[RAW_COLUMNS])
    for i in range(len(DATA)):
        single = pre.transform(DATA[RAW_COLUMNS].iloc[[i]])
        pd.testing.assert_frame_equal(single.reset_index(drop=True),
                                      batch.iloc[[i]].reset_index(drop=True))


def test_unknown_category_is_rejected_not_silently_zeroed():
    row = DATA[RAW_COLUMNS].iloc[[0]].copy()
    row["cp"] = 4
    with pytest.raises(ValueError, match="unknown cp"):
        ClevelandPreprocessor().fit(DATA[RAW_COLUMNS]).transform(row)


def test_categories_cover_training_data():
    for column, categories in CATEGORIES.items():
        assert set(DATA[column].unique()) <= set(categories)


def test_tampered_artifact_is_refused(monkeypatch, tmp_path):
    metadata = (ml.MODELS_DIR / "model_metadata.json").read_text()
    (tmp_path / "model_metadata.json").write_text(metadata)
    (tmp_path / ml.MODEL_METADATA["artifact"]["file"]).write_bytes(b"not the trained model")
    monkeypatch.setattr(ml, "MODELS_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        ml.load_model()


@pytest.mark.parametrize("field,value", [
    ("thalach", "nan"), ("trestbps", "inf"), ("trestbps", "0"), ("ca", "99"),
    ("ca", "1.5"), ("sex", "7"), ("cp", "<script>"), ("age", ""), ("chol", "abc"),
])
def test_validation_rejects_bad_values(field, value):
    _, errors, _ = validate({**VALID_RAW, field: value})
    assert field in errors


def test_validation_warns_outside_training_range():
    _, errors, warnings = validate({**VALID_RAW, "age": "90"})
    assert not errors and "outside the training range" in warnings[0]
