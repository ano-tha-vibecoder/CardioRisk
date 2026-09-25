import base64
import hmac
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import pyotp
import qrcode
import qrcode.image.svg
from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from . import audit
from .extensions import db, login_manager
from .models import User, utcnow

bp = Blueprint("auth", __name__)

# Compared against when the email is unknown so response time does not reveal
# which accounts exist.
_DUMMY_HASH = generate_password_hash("not-a-real-password")
_FAILED = "Sign-in failed. Check your details; repeated failures lock the account temporarily."
MFA_PENDING_SECONDS = 300
MIN_PASSWORD_LENGTH = 12


@login_manager.user_loader
def load_user(user_id: str):
    ident, _, version = user_id.partition(":")
    if not ident.isdigit():
        return None
    user = db.session.get(User, int(ident))
    if user is None or not user.active or str(user.session_version) != version:
        return None
    return user


def _safe_next(target: str | None) -> str | None:
    if not target:
        return None
    parts = urlsplit(target)
    # Only same-site absolute paths: rejects "https://evil", "//evil" and "\\evil".
    if parts.scheme or parts.netloc or not target.startswith("/") or target.startswith(("//", "/\\")):
        return None
    return target


def _register_failure(user: User | None, email: str, reason: str) -> None:
    if user is not None:
        user.failed_logins += 1
        if user.failed_logins >= current_app.config["LOGIN_MAX_FAILURES"]:
            user.locked_until = utcnow() + timedelta(minutes=current_app.config["LOGIN_LOCKOUT_MINUTES"])
            user.failed_logins = 0
            audit.record("account_locked", user, user=user)
    audit.record("login_failed", user, user=user, org_id=user.org_id if user else None,
                 reason=reason, **({} if user else {"email": email[:254]}))
    db.session.commit()


def _complete_login(user: User):
    next_url = _safe_next(session.get("login_next"))
    session.clear()  # new session on privilege change (prevents fixation)
    login_user(user)
    session.permanent = True
    session["login_at"] = utcnow().timestamp()
    user.failed_logins = 0
    user.last_login_at = utcnow()
    audit.record("login", user, user=user)
    db.session.commit()
    return redirect(next_url or url_for("main.dashboard"))


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    if request.method == "GET":
        if nxt := _safe_next(request.args.get("next")):
            # Not "next": Flask-Login's strong session protection deletes that key.
            session["login_next"] = nxt
        return render_template("auth/login.html", expired=request.args.get("expired"))

    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    user = db.session.execute(db.select(User).filter_by(email=email)).scalar_one_or_none()
    now = utcnow()

    if user is None:
        check_password_hash(_DUMMY_HASH, password)
        _register_failure(None, email, "unknown_email")
        return render_template("auth/login.html", error=_FAILED, email=email), 401
    if user.is_locked(now):
        check_password_hash(_DUMMY_HASH, password)
        audit.record("login_failed", user, user=user, reason="locked")
        db.session.commit()
        return render_template("auth/login.html", error=_FAILED, email=email), 401
    if not user.check_password(password) or not user.active:
        _register_failure(user, email, "bad_password" if user.active else "inactive")
        return render_template("auth/login.html", error=_FAILED, email=email), 401

    if user.mfa_secret:
        next_url = session.get("login_next")
        session.clear()
        session["mfa_pending"] = {"user_id": user.id, "at": now.timestamp()}
        if next_url:
            session["login_next"] = next_url
        return redirect(url_for("auth.mfa_verify"))
    return _complete_login(user)


def verify_totp(secret: str, code: str, last_counter: int | None) -> int | None:
    """Return the matched time-step counter, or None. Rejects reused codes."""
    code = code.replace(" ", "")
    if not (code.isdigit() and len(code) == 6):
        return None
    totp = pyotp.TOTP(secret)
    now = totp.timecode(datetime.now(timezone.utc))
    for counter in (now - 1, now, now + 1):
        if last_counter is not None and counter <= last_counter:
            continue
        if hmac.compare_digest(totp.generate_otp(counter), code):
            return counter
    return None


@bp.route("/login/mfa", methods=["GET", "POST"])
def mfa_verify():
    pending = session.get("mfa_pending")
    if not pending or utcnow().timestamp() - pending["at"] > MFA_PENDING_SECONDS:
        session.pop("mfa_pending", None)
        return redirect(url_for("auth.login", expired=1))
    user = db.session.get(User, pending["user_id"])
    if user is None or not user.active or user.is_locked(utcnow()):
        session.clear()
        return redirect(url_for("auth.login"))
    if request.method == "GET":
        return render_template("auth/mfa_verify.html")

    counter = verify_totp(user.mfa_secret, request.form.get("code") or "", user.mfa_last_counter)
    if counter is None:
        _register_failure(user, user.email, "bad_mfa_code")
        return render_template("auth/mfa_verify.html", error="Invalid or already-used code."), 401
    user.mfa_last_counter = counter
    return _complete_login(user)


@bp.route("/logout", methods=["POST"])
def logout():
    if current_user.is_authenticated:
        audit.record("logout", current_user._get_current_object())
        db.session.commit()
    logout_user()
    session.clear()
    return redirect(url_for("auth.login"))


def password_problem(user: User, new: str, confirm: str) -> str | None:
    if len(new) < MIN_PASSWORD_LENGTH:
        return f"Use at least {MIN_PASSWORD_LENGTH} characters."
    if len(new) > 128:
        return "Use at most 128 characters."
    if new != confirm:
        return "The new passwords do not match."
    if user.email.split("@")[0].lower() in new.lower():
        return "The password must not contain your email name."
    if user.check_password(new):
        return "Choose a password different from the current one."
    return None


@bp.route("/account/password", methods=["GET", "POST"])
@login_required
def change_password():
    if request.method == "GET":
        return render_template("auth/change_password.html")
    user = current_user._get_current_object()
    if not user.check_password(request.form.get("current") or ""):
        return render_template("auth/change_password.html", error="Current password is incorrect."), 400
    problem = password_problem(user, request.form.get("new") or "", request.form.get("confirm") or "")
    if problem:
        return render_template("auth/change_password.html", error=problem), 400
    user.set_password(request.form["new"])  # also invalidates other sessions
    user.must_change_password = False
    audit.record("password_changed", user)
    db.session.commit()
    login_user(user)  # re-issue this session with the new session version
    flash("Password updated.")
    return redirect(url_for("main.dashboard"))


def _qr_data_uri(uri: str) -> str:
    svg = qrcode.make(uri, image_factory=qrcode.image.svg.SvgPathImage).to_string()
    return "data:image/svg+xml;base64," + base64.b64encode(svg).decode()


@bp.route("/account/mfa", methods=["GET", "POST"])
@login_required
def mfa_setup():
    user = current_user._get_current_object()
    if user.mfa_secret:
        return render_template("auth/mfa_setup.html", enabled=True)
    secret = session.get("mfa_setup_secret")
    if request.method == "GET" or not secret:
        secret = pyotp.random_base32()
        session["mfa_setup_secret"] = secret
    uri = pyotp.TOTP(secret).provisioning_uri(name=user.email, issuer_name="CardioRisk")
    context = {"secret": secret, "qr": _qr_data_uri(uri), "enabled": False}
    if request.method == "GET":
        return render_template("auth/mfa_setup.html", **context)

    counter = verify_totp(secret, request.form.get("code") or "", None)
    if counter is None:
        return render_template("auth/mfa_setup.html", error="That code did not match. Try the next one.", **context), 400
    user.mfa_secret = secret
    user.mfa_last_counter = counter
    session.pop("mfa_setup_secret", None)
    audit.record("mfa_enabled", user)
    db.session.commit()
    flash("Two-factor authentication is on.")
    return redirect(url_for("main.dashboard"))
