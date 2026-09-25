import os
import secrets
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def database_url(production: bool) -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        if production:
            raise RuntimeError("DATABASE_URL must be set when APP_ENV=production")
        (ROOT / "instance").mkdir(exist_ok=True)
        return f"sqlite:///{ROOT / 'instance' / 'cardiorisk.db'}"
    # Heroku-style URLs use the deprecated "postgres://" scheme.
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def load(production: bool, log) -> dict:
    secret = os.environ.get("SECRET_KEY")
    if not secret:
        if production:
            raise RuntimeError("SECRET_KEY must be set when APP_ENV=production")
        log.warning("SECRET_KEY not set; using an ephemeral key (sessions reset on restart)")
        secret = secrets.token_hex(32)
    return {
        "SECRET_KEY": secret,
        "SQLALCHEMY_DATABASE_URI": database_url(production),
        "SQLALCHEMY_ENGINE_OPTIONS": {"pool_pre_ping": True},
        "MAX_CONTENT_LENGTH": 16 * 1024,
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "SESSION_COOKIE_SECURE": production,
        # Idle timeout: the permanent session cookie is refreshed on every request.
        "PERMANENT_SESSION_LIFETIME": timedelta(minutes=int(os.environ.get("SESSION_IDLE_MINUTES", 30))),
        "SESSION_ABSOLUTE_HOURS": int(os.environ.get("SESSION_ABSOLUTE_HOURS", 12)),
        "WTF_CSRF_TIME_LIMIT": 3600,
        "LOGIN_MAX_FAILURES": 5,
        "LOGIN_LOCKOUT_MINUTES": 15,
        "PRODUCTION": production,
    }
