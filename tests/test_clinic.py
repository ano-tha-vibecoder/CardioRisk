import re
from datetime import date, timedelta

from click.testing import CliRunner

from cardiorisk.extensions import db
from cardiorisk.models import Assessment, AuditEvent, Patient, User, utcnow
from tests.conftest import ROOT, VALID, login, make_org, make_patient, make_user


def _assess(client, patient_id, **overrides):
    return client.post(f"/patients/{patient_id}/assess", data={**VALID, **overrides})


# ── Assessments ─────────────────────────────────────────────────────────────

def test_assessment_is_saved_and_shown(clinic, client):
    patient = clinic["patient"]
    response = _assess(client, patient.id)
    assert response.status_code == 302  # post/redirect/get
    record = db.session.query(Assessment).one()
    assert record.patient_id == patient.id and record.clinician_id == clinic["user"].id
    assert record.model_version and record.inputs["cp"] == 1
    html = client.get(response.headers["Location"]).get_data(as_text=True)
    assert re.search(r'<span class="value">[\d.]+</span>', html)
    assert "ESC 2019" in html and "Lovelace" in html


def test_age_and_sex_come_from_patient_record(clinic, client):
    patient = clinic["patient"]
    _assess(client, patient.id, age="20", sex="0")  # ignored
    record = db.session.query(Assessment).one()
    assert record.inputs["age"] == patient.age_on(date.today())
    assert record.inputs["sex"] == 1


def test_known_golden_result_through_the_app(clinic, client):
    patient = make_patient(clinic["org"], clinic["user"], mrn="G-1",
                           dob=date.today().replace(year=date.today().year - 54) - timedelta(days=1))
    _assess(client, patient.id)
    record = db.session.query(Assessment).filter_by(patient_id=patient.id).one()
    assert round(record.probability * 100, 1) == 15.8
    assert record.ptp == 17  # ESC 2019: male, 50-59, atypical angina


def test_invalid_assessment_is_not_saved(clinic, client):
    response = _assess(client, clinic["patient"].id, ca="9")
    assert response.status_code == 422
    assert 'class="field-error"' in response.get_data(as_text=True)
    assert db.session.query(Assessment).count() == 0


def test_underage_patient_cannot_be_assessed(clinic, client):
    child = make_patient(clinic["org"], clinic["user"], mrn="C-1", dob=date(2015, 1, 1))
    assert _assess(client, child.id).status_code == 422


# ── Patients ────────────────────────────────────────────────────────────────

def test_create_patient_and_duplicate_mrn(clinic, client):
    data = {"mrn": "NEW-1", "first_name": "Grace", "last_name": "Hopper",
            "date_of_birth": "1960-12-09", "sex": "F"}
    assert client.post("/patients/new", data=data).status_code == 302
    assert client.post("/patients/new", data=data).status_code == 422
    assert client.post("/patients/new", data={**data, "mrn": "NEW-2", "date_of_birth": "2999-01-01"}).status_code == 422


def test_same_mrn_allowed_in_different_clinics(clinic, client):
    other = make_org("Clinic B")
    make_patient(other, make_user(other, email="b@b.test"), mrn="SHARED")
    data = {"mrn": "SHARED", "first_name": "A", "last_name": "B", "date_of_birth": "1960-01-01", "sex": "M"}
    assert client.post("/patients/new", data=data).status_code == 302


def test_patient_search(clinic, client):
    make_patient(clinic["org"], clinic["user"], mrn="ZZ-9")
    html = client.get("/patients?q=ZZ-9").get_data(as_text=True)
    assert "ZZ-9" in html and "MRN-1" not in html


def test_edit_patient_is_audited_without_values(clinic, client):
    patient = clinic["patient"]
    data = {"mrn": patient.mrn, "first_name": "Augusta", "last_name": "Lovelace",
            "date_of_birth": "1970-06-15", "sex": "M"}
    assert client.post(f"/patients/{patient.id}/edit", data=data).status_code == 302
    event = db.session.query(AuditEvent).filter_by(action="patient_updated").one()
    assert event.detail == {"fields": ["first_name"]}
    assert "Augusta" not in str(event.detail)


# ── Tenant isolation ────────────────────────────────────────────────────────

def test_other_clinics_data_is_invisible(clinic, client):
    other = make_org("Clinic B")
    other_user = make_user(other, email="b@b.test")
    foreign = make_patient(other, other_user, mrn="FOREIGN")
    record = Assessment(org_id=other.id, patient_id=foreign.id, clinician_id=other_user.id,
                        inputs={}, model_version="x", probability=0.9, risk_band="High",
                        positive=True, contributions=[], warnings=[])
    db.session.add(record)
    db.session.commit()

    assert client.get(f"/patients/{foreign.id}").status_code == 404
    assert client.get(f"/patients/{foreign.id}/edit").status_code == 404
    assert _assess(client, foreign.id).status_code == 404
    assert client.get(f"/assessments/{record.id}").status_code == 404
    assert "0 patients" in client.get("/patients?q=FOREIGN").get_data(as_text=True)
    dashboard = client.get("/dashboard").get_data(as_text=True)
    assert "FOREIGN" not in dashboard
    assert db.session.query(Patient).filter_by(org_id=other.id).count() == 1  # untouched


# ── Dashboard ───────────────────────────────────────────────────────────────

def test_dashboard_counts(clinic, client):
    _assess(client, clinic["patient"].id)
    _assess(client, clinic["patient"].id, ca="3", thal="reversible", exang="1", cp="asymptomatic")
    html = client.get("/dashboard").get_data(as_text=True)
    tiles = re.findall(r'<span class="tile-value">(\d+)</span>', html)
    assert tiles == ["1", "2", "1", "2"]  # patients, 30-day, high-risk, mine
    assert html.count('class="chart-fill"') == 1


def test_weekly_buckets():
    from cardiorisk.main import weekly_counts
    now = utcnow()
    weekly = weekly_counts([now, now - timedelta(days=6), now - timedelta(days=7),
                            now - timedelta(days=83), now - timedelta(days=84)], now)
    assert [c for _, c in weekly][-2:] == [1, 2] and weekly[0][1] == 1
    assert weekly[-1][0] == (now - timedelta(days=7)).date()


# ── Administration ──────────────────────────────────────────────────────────

def test_clinician_cannot_use_admin(clinic, client):
    for path in ("/admin/users", "/admin/settings", "/admin/audit"):
        assert client.get(path).status_code == 403
    assert client.post("/admin/users", data={"email": "x@a.test", "name": "X", "role": "admin"}).status_code == 403


def _admin(client):
    org = make_org()
    admin = make_user(org, email="admin@a.test", role="admin")
    login(client, email="admin@a.test")
    return org, admin


def test_admin_creates_user_with_temporary_password(app, client):
    _admin(client)
    html = client.post("/admin/users", data={"email": "New@A.test", "name": "New Doc", "role": "clinician"}).get_data(as_text=True)
    password = re.search(r'<code class="secret">(.*?)</code>', html).group(1)
    user = db.session.query(User).filter_by(email="new@a.test").one()
    assert user.must_change_password and user.check_password(password)
    duplicate = client.post("/admin/users", data={"email": "new@a.test", "name": "Again", "role": "clinician"})
    assert duplicate.status_code == 422


def test_last_admin_cannot_be_removed(app, client):
    _, admin = _admin(client)
    client.post(f"/admin/users/{admin.id}/deactivate")
    client.post(f"/admin/users/{admin.id}/make_clinician")
    admin = db.session.get(User, admin.id)
    assert admin.active and admin.is_admin


def test_deactivation_ends_sessions(app, client):
    org, _ = _admin(client)
    doc = make_user(org)
    doc_client = app.test_client()
    login(doc_client)
    assert doc_client.get("/dashboard").status_code == 200
    client.post(f"/admin/users/{doc.id}/deactivate")
    assert doc_client.get("/dashboard").status_code == 302


def test_admin_cannot_touch_other_clinics_users(app, client):
    _admin(client)
    other = make_org("Clinic B")
    foreign = make_user(other, email="b@b.test")
    assert client.post(f"/admin/users/{foreign.id}/deactivate").status_code == 404
    assert db.session.get(User, foreign.id).active


def test_audit_log_is_scoped(app, client):
    _admin(client)
    other = make_org("Clinic B")
    make_user(other, email="b@b.test")
    login(app.test_client(), email="b@b.test")
    html = client.get("/admin/audit").get_data(as_text=True)
    assert "admin@a.test" in html and "b@b.test" not in html


# ── Platform ────────────────────────────────────────────────────────────────

def test_security_headers(client):
    response = client.get("/login")
    csp = response.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp
    assert "frame-ancestors 'none'" in csp
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cache-Control"] == "no-store"


def test_templates_have_no_inline_script_or_style():
    for path in (ROOT / "templates").rglob("*.html"):
        html = path.read_text()
        assert " style=" not in html, path.name
        assert "onclick" not in html, path.name
        assert "<script>" not in html, path.name


def test_oversized_request_is_rejected(client):
    assert client.post("/login", data={"email": "x" * 20000}).status_code == 413


def test_healthz_and_landing(clinic, client):
    assert client.get("/healthz").get_json()["status"] == "ok"
    assert client.get("/").status_code == 302  # signed-in users go to the dashboard
    assert client.get("/nope").status_code == 404


def test_landing_public(client):
    html = client.get("/").get_data(as_text=True)
    assert "Sign in" in html and "Cross-validated AUC" in html


def test_cli_create_org(app):
    result = CliRunner().invoke(app.cli, ["create-org", "Clinic C", "--admin-email", "Boss@C.test",
                                          "--admin-name", "Boss"], obj=None)
    assert result.exit_code == 0, result.output
    admin = db.session.query(User).filter_by(email="boss@c.test").one()
    assert admin.is_admin and admin.must_change_password
    password = re.search(r"\(shown once\): (\S+)", result.output).group(1)
    assert admin.check_password(password)


def test_production_requires_secrets(monkeypatch):
    import pytest

    from cardiorisk import create_app
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        create_app()
    monkeypatch.setenv("SECRET_KEY", "x" * 64)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        create_app()
