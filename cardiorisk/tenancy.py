from functools import wraps

from flask import abort
from flask_login import current_user, login_required

from .extensions import db


def scoped(model):
    """Query ``model`` restricted to the signed-in user's organisation."""
    return db.session.query(model).filter(model.org_id == current_user.org_id)


def get_scoped_or_404(model, ident):
    obj = scoped(model).filter(model.id == ident).one_or_none()
    if obj is None:
        # 404 rather than 403: do not reveal that another clinic's id exists.
        abort(404)
    return obj


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapper(*args, **kwargs):
        if not current_user.is_admin:
            abort(403)
        return view(*args, **kwargs)
    return wrapper
