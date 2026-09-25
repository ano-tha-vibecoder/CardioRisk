from flask import request
from flask_login import current_user

from .extensions import db
from .models import AuditEvent


def record(action: str, target=None, *, user=None, org_id=None, **detail) -> None:
    """Add an audit event to the current transaction. Never pass patient data."""
    actor = user if user is not None else (current_user if current_user.is_authenticated else None)
    db.session.add(AuditEvent(
        action=action,
        user_id=actor.id if actor is not None else None,
        org_id=org_id if org_id is not None else (actor.org_id if actor is not None else None),
        target_type=type(target).__name__.lower() if target is not None else None,
        target_id=target.id if target is not None else None,
        ip=request.remote_addr if request else None,
        detail=detail or None,
    ))
