"""Synthetic demo clinic for the public portfolio deployment.

Nothing here is real patient data: names are generated, dates of birth are
derived from ages in the public UCI Cleveland dataset, and clinical inputs
are sampled from that dataset's rows. Seeding is deterministic.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

import pandas as pd

from .config import ROOT
from .extensions import db
from .models import Assessment, AuditEvent, Organization, Patient, User, utcnow
from .patients import create_assessment
from .validation import validate

DEMO_ORG_NAME = "Demo Clinic (synthetic data)"
DEMO_ADMIN = "admin@cardiorisk.demo"
DEMO_CLINICIAN = "clinician@cardiorisk.demo"

FIRST_NAMES = {
    "F": ["Rutendo", "Chipo", "Nyasha", "Thandiwe", "Sibongile", "Tariro", "Rumbidzai",
          "Nokuthula", "Fadzai", "Vimbai", "Ruvimbo", "Nomsa"],
    "M": ["Tendai", "Tatenda", "Farai", "Tafadzwa", "Kudzai", "Sipho", "Nkosana",
          "Tinashe", "Blessing", "Tapiwa", "Mthulisi", "Simba"],
}
LAST_NAMES = ["Moyo", "Ncube", "Dube", "Sibanda", "Ndlovu", "Nyathi", "Gumbo", "Mhlanga",
              "Chirwa", "Banda", "Phiri", "Mapfumo", "Mutasa", "Chikwanha", "Mukandi", "Zulu"]

CODE_TO_FORM = {
    "cp": {0: "typical", 1: "atypical", 2: "non", 3: "asymptomatic"},
    "restecg": {0: "normal", 1: "stt", 2: "lv"},
    "slope": {0: "up", 1: "flat", 2: "down"},
    "thal": {0: "normal", 1: "fixed", 2: "reversible"},
}


def _clear(org: Organization) -> None:
    # Children first: assessments -> patients; audit events -> users.
    for model in (Assessment, Patient, AuditEvent, User):
        db.session.query(model).filter(model.org_id == org.id).delete()


def seed(password: str, patients: int = 36, weeks: int = 12, reset: bool = False, seed_value: int = 7) -> Organization:
    rng = random.Random(seed_value)
    org = db.session.execute(db.select(Organization).filter_by(name=DEMO_ORG_NAME)).scalar_one_or_none()
    if org is not None and not reset:
        raise RuntimeError("demo clinic already exists; pass reset=True to recreate it")
    if org is None:
        org = Organization(name=DEMO_ORG_NAME)
        db.session.add(org)
        db.session.flush()
    else:
        _clear(org)
        org.require_mfa = False

    users = []
    for email, name, role in ((DEMO_ADMIN, "Dr Demo Admin", "admin"),
                              (DEMO_CLINICIAN, "Dr Demo Clinician", "clinician")):
        user = User(org_id=org.id, email=email, name=name, role=role, must_change_password=False)
        user.set_password(password)
        db.session.add(user)
        users.append(user)
    db.session.flush()

    rows = pd.read_csv(ROOT / "data/heart_cleveland_upload.csv").to_dict("records")
    rng.shuffle(rows)
    now = utcnow()
    today = date.today()
    for i, row in enumerate(rows[:patients]):
        sex = "M" if row["sex"] == 1 else "F"
        age = int(row["age"])
        # Birthday passed 1-360 days ago, so the patient is exactly `age` today
        # (day capped at 28 so 29 February cannot produce an invalid date).
        dob = date(today.year - age, today.month, min(today.day, 28)) - timedelta(days=rng.randint(1, 360))
        patient = Patient(org_id=org.id, created_by_id=users[i % 2].id, mrn=f"DEMO-{1001 + i}",
                          first_name=rng.choice(FIRST_NAMES[sex]), last_name=rng.choice(LAST_NAMES),
                          date_of_birth=dob, sex=sex,
                          created_at=now - timedelta(days=7 * weeks, hours=rng.randint(0, 48)))
        db.session.add(patient)
        db.session.flush()

        form = {k: str(row[k]) for k in ("trestbps", "chol", "fbs", "thalach", "exang", "oldpeak", "ca")}
        form.update({k: CODE_TO_FORM[k][int(row[k])] for k in CODE_TO_FORM})
        form.update(age=str(patient.age_on(today)), sex="1" if sex == "M" else "0")
        values, errors, warnings = validate(form)
        if errors:  # every Cleveland row is inside the hard limits; fail loudly if not
            raise RuntimeError(f"demo row failed validation: {sorted(errors)}")
        # Weighted towards recent weeks so the dashboard looks like a live clinic.
        days_ago = min(int(rng.expovariate(1 / 25)), 7 * weeks - 1)
        created = now - timedelta(days=days_ago, hours=rng.randint(0, 9), minutes=rng.randint(0, 59))
        create_assessment(patient, values, warnings, rng.choice(users), created_at=created)
    db.session.commit()
    return org
