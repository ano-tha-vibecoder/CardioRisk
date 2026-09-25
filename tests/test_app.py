import re

import numpy as np
import pandas as pd
import pytest

import app as app_module
import schema
import train
from preprocessing import CATEGORIES, RAW_COLUMNS, ClevelandPreprocessor
from tests.conftest import ROOT, VALID

DATA = pd.read_csv(ROOT / "data/heart_cleveland_upload.csv")


def _probability(html):
    return re.search(r'<span class="value">(.*?)</span>', html).group(1)


# ── Prediction ──────────────────────────────────────────────────────────────

def test_representative_assessment_golden_output(client):
    response = client.post("/assess", data=VALID)
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert _probability(html) == "15.8"
    assert "Low Risk" in html
    labels = re.findall(r'<span class="shap-label">(.*?): <span class="shap-shown">(.*?)</span>', html)
    assert labels[:3] == [("Major vessels coloured", "0"), ("Thallium test", "Normal"),
                          ("ST slope", "Upsloping")]


def test_shipped_artifact_matches_training_code():
    """Guards against a stale artifact that train.py would no longer produce."""
    X, y = DATA[RAW_COLUMNS], DATA["condition"]
    fresh = train.build_pipeline().fit(X, y)
    np.testing.assert_allclose(fresh.predict_proba(X), app_module.PIPELINE.predict_proba(X))


def test_contributions_match_shap_linear_explainer():
    shap = pytest.importorskip("shap")
    pipeline = app_module.PIPELINE
    background = pipeline[:-1].transform(DATA[RAW_COLUMNS])
    # shap's default masker subsamples 100 background rows; use them all.
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.LinearExplainer(pipeline[-1], masker)
    row = DATA[RAW_COLUMNS].iloc[[5]]
    expected = explainer.shap_values(pipeline[:-1].transform(row))[0]
    manual = pipeline[-1].coef_[0] * pipeline[:-1].transform(row)[0]
    np.testing.assert_allclose(manual, expected, atol=1e-9)


def test_grouped_contributions_are_exact():
    values, errors, _ = schema.validate(VALID)
    assert not errors
    features = app_module.PIPELINE[:-1].transform(pd.DataFrame([values], columns=RAW_COLUMNS))
    model = app_module.PIPELINE[-1]
    grouped = app_module.explain(features)
    assert set(grouped) == {*RAW_COLUMNS, "hr_bp_ratio"}
    # SHAP efficiency: contributions sum to f(x) - E[f(x)] in log-odds.
    assert sum(grouped.values()) == pytest.approx(float(model.coef_[0] @ features[0]))


# ── Preprocessing ───────────────────────────────────────────────────────────

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


# ── Validation ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("field,value", [
    ("thalach", "nan"), ("trestbps", "inf"), ("trestbps", "0"), ("ca", "99"),
    ("ca", "1.5"), ("sex", "7"), ("cp", "<script>"), ("age", ""), ("chol", "abc"),
])
def test_invalid_input_is_rejected_without_leaking_internals(client, field, value):
    response = client.post("/assess", data={**VALID, field: value})
    html = response.get_data(as_text=True)
    assert response.status_code == 422
    assert 'class="field-error"' in html
    assert "Traceback" not in html and "sklearn" not in html
    assert "<script>" not in html


def test_missing_field_is_a_validation_error_not_a_crash(client):
    data = dict(VALID)
    data.pop("age")
    assert client.post("/assess", data=data).status_code == 422


def test_out_of_training_range_value_warns(client):
    html = client.post("/assess", data={**VALID, "age": "90"}).get_data(as_text=True)
    assert "outside the training range" in html


def test_form_values_are_preserved_on_error(client):
    html = client.post("/assess", data={**VALID, "ca": "9"}).get_data(as_text=True)
    assert 'value="245"' in html
    assert '<option value="atypical" selected>' in html


# ── Security ────────────────────────────────────────────────────────────────

def test_post_without_csrf_token_is_rejected(csrf_client):
    response = csrf_client.post("/assess", data=VALID)
    assert response.status_code == 400
    assert '<span class="value">' not in response.get_data(as_text=True)


def test_post_with_csrf_token_succeeds(csrf_client):
    form = csrf_client.get("/assess").get_data(as_text=True)
    token = re.search(r'name="csrf_token" value="(.*?)"', form).group(1)
    assert csrf_client.post("/assess", data={**VALID, "csrf_token": token}).status_code == 200


def test_security_headers(client):
    response = client.get("/assess")
    csp = response.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp
    assert "frame-ancestors 'none'" in csp
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cache-Control"] == "no-store"


def test_templates_have_no_inline_script_or_style():
    for path in (ROOT / "templates").glob("*.html"):
        html = path.read_text()
        assert " style=" not in html, path.name
        assert "onclick" not in html, path.name
        assert "<script>" not in html, path.name


def test_oversized_request_is_rejected(client):
    assert client.post("/assess", data={**VALID, "pad": "x" * 20000}).status_code == 413


def test_tampered_artifact_is_refused(monkeypatch, tmp_path):
    metadata = (app_module.MODELS_DIR / "model_metadata.json").read_text()
    (tmp_path / "model_metadata.json").write_text(metadata)
    name = app_module.MODEL_METADATA["artifact"]["file"]
    (tmp_path / name).write_bytes(b"not the trained model")
    monkeypatch.setattr(app_module, "MODELS_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        app_module.load_model()


def test_production_requires_secret_key(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        app_module.create_app()


# ── Pages ───────────────────────────────────────────────────────────────────

def test_landing_shows_metrics_from_metadata(client):
    html = client.get("/").get_data(as_text=True)
    auc = app_module.MODEL_METADATA["evaluation"]["metrics"]["auc"]["mean"]
    assert f"{auc:.2f}" in html
    assert f'>{len(app_module.FEATURES)}<' in html


def test_healthz(client):
    body = client.get("/healthz").get_json()
    assert body == {"status": "ok", "model_version": app_module.MODEL_METADATA["model_version"]}


def test_unknown_page_returns_404(client):
    assert client.get("/nope").status_code == 404
