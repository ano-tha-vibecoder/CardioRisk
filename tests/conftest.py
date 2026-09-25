import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import app as app_module  # noqa: E402

VALID = {
    "age": "54", "sex": "1", "trestbps": "130", "chol": "245",
    "fbs": "0", "thalach": "150", "exang": "0", "oldpeak": "1.5",
    "ca": "0", "cp": "atypical", "restecg": "normal",
    "thal": "normal", "slope": "up",
}


@pytest.fixture
def client():
    flask_app = app_module.create_app({"TESTING": True, "WTF_CSRF_ENABLED": False})
    return flask_app.test_client()


@pytest.fixture
def csrf_client():
    return app_module.create_app({"TESTING": True}).test_client()
