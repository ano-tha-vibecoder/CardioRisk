# CardioRisk

Multi-clinic web app for estimating the probability of angiographically
significant coronary artery disease (CAD). Each assessment is stored in the
patient's record and shows:

* an **ML estimate** (logistic regression) with per-variable SHAP explanations, and
* the **ESC 2019 guideline pre-test probability** (age, sex, chest-pain type),

side by side. A heuristic flag highlights gross disagreement between the two.

> Decision-support prototype. Not externally validated, not a medical device,
> not cleared by any regulator. See "Before storing real patient data" below.

## Product overview

| Area | What it does |
|---|---|
| Clinics (tenants) | Every patient, assessment, user and audit event belongs to one organisation. All queries are scoped to the signed-in user's organisation; ids from other clinics return 404. |
| Roles | `admin` (users, security settings, audit log, plus clinical work) and `clinician`. |
| Sign-in | Email + password (scrypt hashes), lockout after 5 failures for 15 min, generic error messages, optional TOTP two-factor that admins can require clinic-wide, forced change of admin-issued temporary passwords. |
| Sessions | 30 min idle timeout, 12 h absolute limit, rotated on login, invalidated on password change, deactivation or 2FA reset. |
| Patients | Create, edit, search by name/MRN. MRN unique per clinic. |
| Assessments | Immutable records storing the inputs, model version, ML result, SHAP contributions and guideline score. Age and sex come from the patient record. |
| Dashboard | Patient count, 30-day volume, high-risk count, risk-band distribution, weekly volume (12 weeks), recent assessments. |
| Audit log | Sign-ins (and failures), record views, creations, edits and admin actions, with user, IP and time. Never stores clinical values. |

## The model — and its limits

| | |
|---|---|
| Data | UCI Cleveland heart disease, 297 complete cases, 46% prevalence |
| Target | >50% diameter narrowing on coronary angiography |
| Model | L2-regularised logistic regression on 19 encoded features |
| Evaluation | Repeated stratified 5-fold CV × 10 of the full pipeline |
| AUC | 0.91 ± 0.03 · sensitivity 78% · specificity 88% (threshold 0.5) |

* **Referral population.** Every patient in the cohort had already been
  referred for angiography, in 1980s Cleveland. Probabilities will not carry
  over to a screening population, and are probably miscalibrated elsewhere.
* **Diagnostic, not prognostic.** `ca` (fluoroscopy) and `thal` (thallium
  imaging) are late work-up results. This is not a 10-year risk score.
* **Small sample.** 297 patients; the metric standard deviations are wide.
* **Guideline table.** `cardiorisk/scores.py` reproduces ESC 2019 Table 5,
  which the 2024 ESC chronic coronary syndrome guideline replaced with a
  risk-factor-weighted model. Before clinical use, clinical governance must
  confirm which one to use and check the values against the published
  source. They are pinned in `tests/test_scores.py`.

## Run locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
export FLASK_APP=app
flask db upgrade                       # SQLite in instance/ by default
flask create-org "Demo Clinic" --admin-email you@example.com --admin-name "Your Name"
flask run                              # sign in with the printed temporary password
pytest                                 # 69 tests; TEST_DATABASE_URL=postgresql+psycopg://... to use Postgres
```

Retrain the model (deterministic; rewrites `models/`): `python train.py`.

## Deploy

`Procfile` runs migrations in the release phase, then gunicorn.

| Variable | Required | Purpose |
|---|---|---|
| `APP_ENV=production` | yes | Secure cookies, trusts one proxy hop for HTTPS/IP, refuses to start without the two below |
| `SECRET_KEY` | yes | Signs session and CSRF cookies. `python -c "import secrets; print(secrets.token_hex(32))"` |
| `DATABASE_URL` | yes | Postgres URL (`postgres://` and `postgresql://` are accepted) |
| `SESSION_IDLE_MINUTES` | no | Default 30 |
| `SESSION_ABSOLUTE_HOURS` | no | Default 12 |
| `LOG_LEVEL` | no | Default `INFO` |

Clinics are onboarded by the platform operator with `flask create-org`;
there is no self-service sign-up. `GET /healthz` checks the database and
returns the model version.

## Security controls

* Tenant scoping through one helper (`cardiorisk/tenancy.py`), with tests
  proving cross-clinic reads, edits and admin actions return 404.
* CSRF on every form (including login and logout), a strict CSP (no inline
  script or style), `X-Frame-Options: DENY`, `nosniff`, `no-referrer`, HSTS
  behind TLS, `Cache-Control: no-store`, 16 KB request limit.
* TOTP codes cannot be reused (last accepted time step is stored).
* Open-redirect protection on the post-login `next` URL.
* Server-side validation of every clinical input; prediction errors are
  logged without the submitted values.
* The model artifact's SHA-256 is checked before unpickling. That catches
  corruption, but not an attacker who can write to both the artifact and
  its metadata, so the deploy pipeline must be protected.

## Before storing real patient data

The code is built with these requirements in mind, but none of them is met
by code alone. Which ones apply depends on your jurisdiction (HIPAA,
UK GDPR/DSPT, EU GDPR/MDR):

* **Hosting and contracts.** A hosting provider and database that sign a
  BAA (US) or data processing agreement; encryption at rest; backups with
  tested restores; region pinning if required.
* **Medical device status.** Patient-specific diagnostic output for
  clinicians is likely regulated software (FDA SaMD / EU MDR Class IIa+).
  Get regulatory advice before selling it.
* **Not built yet.** Rate limiting per IP (lockout is per account only),
  SSO/SAML for enterprise customers, clinic-local time zones, data export
  and deletion workflows, retention policy, a break-glass procedure,
  encryption of TOTP secrets at the application level, external
  monitoring and alerting, billing.

## Layout

```
app.py                  WSGI entry point
cardiorisk/
  __init__.py           App factory, error pages
  config.py             Environment-driven configuration
  models.py             Organization, User, Patient, Assessment, AuditEvent
  tenancy.py            Org-scoped queries, admin guard
  security.py           Session policy gate, security headers
  auth.py               Sign-in, 2FA, password change, sign-out
  main.py               Landing, dashboard, health check
  patients.py           Patients and assessments
  admin.py              Users, clinic security settings, audit log
  ml.py                 Model loading, prediction, SHAP
  scores.py             ESC 2019 pre-test probability
  validation.py         Clinical input limits and validation
  audit.py, cli.py
preprocessing.py        Shared raw -> feature encoding (pickled with the model)
train.py                CV evaluation + production fit + metadata
migrations/             Alembic migrations
models/                 cardio_pipeline.joblib, model_metadata.json
```
