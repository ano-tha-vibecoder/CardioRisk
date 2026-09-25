import os
import re
import sys
from datetime import date
from pathlib import Path

import pytest
from flask import g

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from cardiorisk import create_app  # noqa: E402
from cardiorisk.extensions import db  # noqa: E402
from cardiorisk.models import Organization, Patient, User  # noqa: E402

PASSWORD = "correct horse battery staple"
# Set TEST_DATABASE_URL=postgresql+psycopg://... to run the suite against Postgres.
TEST_DB = os.environ.get("TEST_DATABASE_URL", "sqlite://")

VALID = {
    "trestbps": "130", "chol": "245", "fbs": "0", "thalach": "150", "exang": "0",
    "oldpeak": "1.5", "ca": "0", "cp": "atypical", "restecg": "normal",
    "thal": "normal", "slope": "up",
}
# The same inputs in the raw model format (age 54, male), for ML-only tests.
VALID_RAW = {**VALID, "age": "54", "sex": "1"}


def _isolate_requests(app):
    # The fixture keeps one app context open so tests can use db.session, and
    # Flask's test client reuses it for every request. Drop per-request state
    # that would otherwise leak between requests (as it never does in production).
    def _fresh_request_state():
        g.pop("_login_user", None)

    # Must run before the app's own hooks, which read current_user.
    app.before_request_funcs.setdefault(None, []).insert(0, _fresh_request_state)


@pytest.fixture
def app():
    app = create_app({"TESTING": True, "WTF_CSRF_ENABLED": False,
                      "SQLALCHEMY_DATABASE_URI": TEST_DB})
    _isolate_requests(app)
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def csrf_app():
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": TEST_DB})
    _isolate_requests(app)
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


def make_org(name="Clinic A", **kwargs):
    org = Organization(name=name, **kwargs)
    db.session.add(org)
    db.session.commit()
    return org


def make_user(org, email="doc@a.test", role="clinician", password=PASSWORD, **kwargs):
    user = User(org_id=org.id, email=email, name=email.split("@")[0].title(), role=role,
                must_change_password=kwargs.pop("must_change_password", False), **kwargs)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return user


def make_patient(org, user, mrn="MRN-1", dob=date(1970, 6, 15), sex="M"):
    patient = Patient(org_id=org.id, created_by_id=user.id, mrn=mrn, first_name="Ada",
                      last_name="Lovelace", date_of_birth=dob, sex=sex)
    db.session.add(patient)
    db.session.commit()
    return patient


def login(client, email="doc@a.test", password=PASSWORD):
    return client.post("/login", data={"email": email, "password": password})


def csrf_token(html):
    return re.search(r'name="csrf_token" value="(.*?)"', html).group(1)


@pytest.fixture
def clinic(app, client):
    """An org with a signed-in clinician and one patient."""
    org = make_org()
    user = make_user(org)
    patient = make_patient(org, user)
    assert login(client).status_code == 302
    return {"org": org, "user": user, "patient": patient}
