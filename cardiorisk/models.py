"""Database models.

Every clinical row carries ``org_id``. Views must go through
``scoped(Model)`` (see ``tenancy.py``) so one clinic can never read another
clinic's data.
"""
from __future__ import annotations

import secrets
from datetime import date, datetime, timezone

from flask_login import UserMixin
from sqlalchemy import JSON, UniqueConstraint
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db

ROLES = ("admin", "clinician")
DEFAULT_TIMEZONE = "Africa/Harare"


def utcnow() -> datetime:
    # Stored as naive UTC so SQLite and Postgres behave identically.
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Organization(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    require_mfa = db.Column(db.Boolean, nullable=False, default=False)
    # IANA zone used to display times; everything is stored in UTC.
    timezone = db.Column(db.String(64), nullable=False, default=DEFAULT_TIMEZONE,
                         server_default=DEFAULT_TIMEZONE)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    users = db.relationship("User", back_populates="organization", lazy="dynamic")


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    org_id = db.Column(db.Integer, db.ForeignKey("organization.id"), nullable=False, index=True)
    email = db.Column(db.String(254), nullable=False, unique=True)
    name = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="clinician")
    password_hash = db.Column(db.String(255), nullable=False)
    active = db.Column(db.Boolean, nullable=False, default=True)
    must_change_password = db.Column(db.Boolean, nullable=False, default=True)
    failed_logins = db.Column(db.Integer, nullable=False, default=0)
    locked_until = db.Column(db.DateTime)
    mfa_secret = db.Column(db.String(64))
    mfa_last_counter = db.Column(db.BigInteger)
    # Bumped on password change, deactivation or "sign out everywhere";
    # embedded in the session id so existing sessions stop validating.
    session_version = db.Column(db.Integer, nullable=False, default=1)
    last_login_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    organization = db.relationship("Organization", back_populates="users")

    @property
    def is_active(self) -> bool:
        return self.active

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def get_id(self) -> str:
        return f"{self.id}:{self.session_version}"

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)
        self.session_version = (self.session_version or 0) + 1

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def is_locked(self, now: datetime) -> bool:
        return self.locked_until is not None and self.locked_until > now


class Patient(db.Model):
    __table_args__ = (UniqueConstraint("org_id", "mrn", name="uq_patient_org_mrn"),)

    id = db.Column(db.Integer, primary_key=True)
    org_id = db.Column(db.Integer, db.ForeignKey("organization.id"), nullable=False, index=True)
    mrn = db.Column(db.String(64), nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    sex = db.Column(db.String(1), nullable=False)  # "F" / "M" as recorded in the model's data
    created_by_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    assessments = db.relationship("Assessment", back_populates="patient", lazy="dynamic",
                                  order_by="Assessment.created_at.desc()")

    @property
    def full_name(self) -> str:
        return f"{self.last_name}, {self.first_name}"

    def age_on(self, day: date) -> int:
        dob = self.date_of_birth
        return day.year - dob.year - ((day.month, day.day) < (dob.month, dob.day))


class Assessment(db.Model):
    """An immutable record of one risk assessment, as shown to the clinician."""

    id = db.Column(db.Integer, primary_key=True)
    org_id = db.Column(db.Integer, db.ForeignKey("organization.id"), nullable=False, index=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=False, index=True)
    clinician_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    inputs = db.Column(JSON, nullable=False)
    model_version = db.Column(db.String(20), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    risk_band = db.Column(db.String(10), nullable=False)
    positive = db.Column(db.Boolean, nullable=False)
    contributions = db.Column(JSON, nullable=False)
    warnings = db.Column(JSON, nullable=False, default=list)
    ptp = db.Column(db.Float)  # guideline pre-test probability, None when not applicable
    ptp_note = db.Column(db.String(200))

    patient = db.relationship("Patient", back_populates="assessments")
    clinician = db.relationship("User")


class AuditEvent(db.Model):
    """Append-only access log. ``detail`` must never contain patient data."""

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    org_id = db.Column(db.Integer, db.ForeignKey("organization.id"), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    action = db.Column(db.String(50), nullable=False)
    target_type = db.Column(db.String(30))
    target_id = db.Column(db.Integer)
    ip = db.Column(db.String(64))
    detail = db.Column(JSON)

    user = db.relationship("User")


def temporary_password() -> str:
    return secrets.token_urlsafe(12)
