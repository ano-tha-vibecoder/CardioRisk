import re
from datetime import datetime

from click.testing import CliRunner

from cardiorisk import demo
from cardiorisk.extensions import db
from cardiorisk.models import Assessment, Organization, Patient, User
from tests.conftest import ROOT, login, make_org, make_user


# ── Time zones ──────────────────────────────────────────────────────────────

def test_times_are_shown_in_clinic_time_zone(clinic, client):
    record = Assessment(org_id=clinic["org"].id, patient_id=clinic["patient"].id,
                        clinician_id=clinic["user"].id, created_at=datetime(2026, 3, 1, 22, 30),
                        inputs={}, model_version="x", probability=0.2, risk_band="Low",
                        positive=False, contributions=[], warnings=[])
    db.session.add(record)
    db.session.commit()
    html = client.get(f"/patients/{clinic['patient'].id}").get_data(as_text=True)
    assert clinic["org"].timezone == "Africa/Harare"
    assert "2026-03-02 00:30" in html  # 22:30 UTC is 00:30 next day in Harare (UTC+2)
    assert "Times are Africa/Harare" in html


def test_admin_can_change_time_zone_but_not_to_arbitrary_values(app, client):
    org = make_org()
    make_user(org, email="admin@a.test", role="admin")
    login(client, email="admin@a.test")
    client.post("/admin/settings", data={"timezone": "Africa/Johannesburg"})
    assert db.session.get(Organization, org.id).timezone == "Africa/Johannesburg"
    client.post("/admin/settings", data={"timezone": "Mars/Olympus"})
    assert db.session.get(Organization, org.id).timezone == "Africa/Johannesburg"


# ── Demo ────────────────────────────────────────────────────────────────────

def test_seed_demo_is_deterministic_and_resettable(app):
    runner = CliRunner()
    first = runner.invoke(app.cli, ["seed-demo", "--patients", "10"])
    assert first.exit_code == 0, first.output
    org = db.session.execute(db.select(Organization).filter_by(name=demo.DEMO_ORG_NAME)).scalar_one()
    names = sorted(p.full_name for p in db.session.query(Patient).filter_by(org_id=org.id))
    probabilities = sorted(a.probability for a in db.session.query(Assessment).filter_by(org_id=org.id))
    assert len(names) == len(probabilities) == 10

    assert runner.invoke(app.cli, ["seed-demo"]).exit_code != 0  # refuses to double-seed
    idle = runner.invoke(app.cli, ["seed-demo", "--if-missing"])
    assert idle.exit_code == 0 and "nothing to do" in idle.output
    again = runner.invoke(app.cli, ["seed-demo", "--reset", "--patients", "10"])
    assert again.exit_code == 0, again.output
    db.session.expire_all()
    assert sorted(p.full_name for p in db.session.query(Patient).filter_by(org_id=org.id)) == names
    assert sorted(a.probability for a in db.session.query(Assessment).filter_by(org_id=org.id)) == probabilities
    assert db.session.query(User).filter_by(org_id=org.id).count() == 2


def test_demo_seed_does_not_touch_other_clinics(app):
    other = make_org("Real Clinic")
    make_user(other, email="real@a.test")
    demo.seed("pw-for-tests", patients=3)
    demo.seed("pw-for-tests", patients=3, reset=True)
    assert db.session.query(User).filter_by(org_id=other.id).count() == 1


def _demo_client(app):
    app.config["DEMO_MODE"] = True
    demo.seed(app.config["DEMO_PASSWORD"], patients=3)
    return app.test_client()


def test_demo_login_page_shows_credentials_and_banner(app):
    client = _demo_client(app)
    html = client.get("/login").get_data(as_text=True)
    assert demo.DEMO_ADMIN in html and app.config["DEMO_PASSWORD"] in html
    assert "synthetic data" in html


def test_demo_blocks_account_changes(app):
    client = _demo_client(app)
    login(client, email=demo.DEMO_ADMIN, password=app.config["DEMO_PASSWORD"])
    pw = app.config["DEMO_PASSWORD"]
    client.post("/account/password", data={"current": pw, "new": "x" * 20, "confirm": "x" * 20})
    admin = db.session.execute(db.select(User).filter_by(email=demo.DEMO_ADMIN)).scalar_one()
    assert admin.check_password(pw)
    clinician = db.session.execute(db.select(User).filter_by(email=demo.DEMO_CLINICIAN)).scalar_one()
    client.post(f"/admin/users/{clinician.id}/deactivate")
    client.post("/admin/users", data={"email": "new@a.test", "name": "New", "role": "admin"})
    client.post("/admin/settings", data={"require_mfa": "1", "timezone": "UTC"})
    db.session.expire_all()
    assert clinician.active and not admin.organization.require_mfa
    assert db.session.execute(db.select(User).filter_by(email="new@a.test")).scalar_one_or_none() is None


def test_demo_accounts_cannot_be_locked_out(app):
    client = _demo_client(app)
    for _ in range(app.config["LOGIN_MAX_FAILURES"] + 2):
        login(client, email=demo.DEMO_CLINICIAN, password="wrong")
    assert login(client, email=demo.DEMO_CLINICIAN, password=app.config["DEMO_PASSWORD"]).status_code == 302


def test_demo_still_allows_clinical_workflow(app):
    client = _demo_client(app)
    login(client, email=demo.DEMO_CLINICIAN, password=app.config["DEMO_PASSWORD"])
    data = {"mrn": "TRY-1", "first_name": "Test", "last_name": "Visitor",
            "date_of_birth": "1965-01-01", "sex": "F"}
    assert client.post("/patients/new", data=data).status_code == 302


# ── No third-party requests ─────────────────────────────────────────────────

def test_no_external_resources():
    for path in [*(ROOT / "templates").rglob("*.html"), ROOT / "static/css/style.css"]:
        text = path.read_text()
        assert not re.search(r"(src|href)=[\"']https?://", text), path.name
        assert "@import" not in text and "googleapis" not in text, path.name


def test_csp_allows_only_self_for_fonts_and_styles(client):
    csp = client.get("/login").headers["Content-Security-Policy"]
    assert "font-src 'self'" in csp and "style-src 'self';" in csp and "https://" not in csp
