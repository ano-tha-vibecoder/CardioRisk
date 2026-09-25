# CardioRisk

Flask web app that estimates the probability of angiographically significant
coronary artery disease (CAD) from 13 clinical inputs, with per-variable SHAP
explanations.

> Decision-support prototype. Not externally validated, not a medical device,
> not cleared by any regulator.

## What the model is — and is not

| | |
|---|---|
| Data | UCI Cleveland heart disease, 297 complete cases, 46% prevalence |
| Target | >50% diameter narrowing on coronary angiography |
| Model | L2-regularised logistic regression on 19 encoded features |
| Evaluation | Repeated stratified 5-fold CV × 10 of the full pipeline |
| AUC | 0.91 ± 0.03 · sensitivity 78% · specificity 88% (threshold 0.5) |

Limitations to keep in mind:

* **Referral population.** Every patient in the cohort was already sent for
  angiography in 1980s Cleveland. Probabilities do not transfer to a general
  screening population and are likely miscalibrated elsewhere.
* **Diagnostic, not prognostic.** Inputs include `ca` (fluoroscopy) and
  `thal` (thallium stress imaging), which are only available late in a
  cardiac work-up. This is not a 10-year risk score like PCE, SCORE2 or QRISK.
* **Small sample.** 297 patients; metric standard deviations are wide.
* **Sex is binary** in the source data.
* Inputs outside the training range are accepted within physiological limits
  but flagged as extrapolation.

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest
python app.py            # http://127.0.0.1:5000
```

Retrain (deterministic; rewrites `models/`):

```bash
python train.py
```

## Deploy

`Procfile` runs gunicorn. Required environment:

| Variable | Purpose |
|---|---|
| `APP_ENV=production` | Enables secure cookies, trusts one proxy hop for HTTPS detection, requires `SECRET_KEY` |
| `SECRET_KEY` | Signs the session/CSRF cookie. Generate with `python -c "import secrets; print(secrets.token_hex(32))"` |

`GET /healthz` returns the status and the model version.

## Security notes

* All inputs are validated server-side (`schema.py`); the model never sees
  NaN, infinity or out-of-domain categories.
* CSRF protection (Flask-WTF), strict CSP (no inline script or style),
  `X-Frame-Options`, `nosniff`, `Referrer-Policy`, HSTS behind TLS,
  `Cache-Control: no-store` on all pages, 16 KB request limit.
* The model artifact is a joblib pickle, which runs code when it is loaded. The
  app refuses to load it unless its SHA-256 matches `model_metadata.json`.
  This catches corruption and mismatched artifacts. It does **not** stop an
  attacker who can write to both files, so protect the deploy pipeline.
* Submitted values are never logged.

## Layout

```
app.py            Flask app, prediction + explanation
schema.py         Input limits, training ranges, validation
preprocessing.py  Shared raw -> feature encoding (training and inference)
train.py          CV evaluation + production fit + metadata
models/           cardio_pipeline.joblib, model_metadata.json
heart_disease.ipynb  Exploratory only (see note in first cell)
```
