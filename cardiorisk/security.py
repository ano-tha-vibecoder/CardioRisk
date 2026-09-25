from datetime import timedelta

from flask import current_app, redirect, request, session, url_for
from flask_login import current_user, logout_user

from .models import utcnow

CSP = (
    "default-src 'self'; "
    "style-src 'self' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; "
    "img-src 'self' data:; script-src 'self'; "
    "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)

# Endpoints a signed-in user may reach while a forced step is pending.
_GATE_EXEMPT = {"static", "auth.logout", "auth.change_password", "auth.mfa_setup", "main.healthz"}


def init_app(app):
    @app.before_request
    def enforce_session_policy():
        if not current_user.is_authenticated:
            return None
        login_at = session.get("login_at")
        limit = timedelta(hours=current_app.config["SESSION_ABSOLUTE_HOURS"])
        if login_at is None or utcnow().timestamp() - login_at > limit.total_seconds():
            logout_user()
            session.clear()
            return redirect(url_for("auth.login", expired=1))
        if request.endpoint in _GATE_EXEMPT:
            return None
        if current_user.must_change_password:
            return redirect(url_for("auth.change_password"))
        if current_user.organization.require_mfa and not current_user.mfa_secret:
            return redirect(url_for("auth.mfa_setup"))
        return None

    @app.after_request
    def security_headers(response):
        response.headers["Content-Security-Policy"] = CSP
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
