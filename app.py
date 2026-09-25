import hashlib
import json
import logging
import os
import secrets
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from flask_wtf.csrf import CSRFError, CSRFProtect
from werkzeug.middleware.proxy_fix import ProxyFix

import schema
from preprocessing import CATEGORIES, RAW_COLUMNS
from preprocessing import ClevelandPreprocessor  # noqa: F401  (must be importable to unpickle)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("cardiorisk")


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    production = os.environ.get("APP_ENV") == "production"
    if production:
        # Behind a TLS-terminating proxy (Heroku, Render, a load balancer):
        # trust one hop so request.is_secure and remote_addr are correct.
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    secret = os.environ.get("SECRET_KEY")
    if not secret:
        if production:
            raise RuntimeError("SECRET_KEY must be set when APP_ENV=production")
        log.warning("SECRET_KEY not set; using an ephemeral key (sessions reset on restart)")
        secret = secrets.token_hex(32)
    app.config.update(
        SECRET_KEY=secret,
        MAX_CONTENT_LENGTH=16 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=production,
        WTF_CSRF_TIME_LIMIT=3600,
    )
    app.config.update(config or {})
    CSRFProtect(app)
    register_routes(app)
    return app


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
        "prob": round(prob * 100, 1),
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


def register_routes(app: Flask) -> None:
    @app.context_processor
    def inject_metadata():
        return {"model_metadata": MODEL_METADATA}

    @app.after_request
    def security_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "style-src 'self' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; "
            "img-src 'self' data:; script-src 'self'; "
            "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if request.is_secure:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        if request.endpoint != "static":
            # Pages may contain patient data: never cache them.
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/")
    def landing():
        return render_template("landing.html")

    @app.route("/healthz")
    def healthz():
        return {"status": "ok", "model_version": MODEL_METADATA["model_version"]}

    @app.route("/assess", methods=["GET", "POST"])
    def assess():
        if request.method == "GET":
            return render_template("assess.html", form={}, errors={})

        values, errors, warnings = schema.validate(request.form)
        if errors:
            return render_template("assess.html", form=request.form, errors=errors), 422
        try:
            result = predict(values)
        except Exception:
            log.exception("prediction failed")  # never log the submitted values (PHI)
            return render_template(
                "assess.html", form=request.form, errors={},
                error="The assessment could not be completed. Please try again.",
            ), 500
        return render_template("result.html", result=result, warnings=warnings)

    @app.errorhandler(CSRFError)
    def csrf_error(_):
        return render_template(
            "assess.html", form={}, errors={},
            error="Your session expired. Please re-enter the form.",
        ), 400

    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", code=404, message="Page not found."), 404

    @app.errorhandler(413)
    def too_large(_):
        return render_template("error.html", code=413, message="Request too large."), 413


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False)
