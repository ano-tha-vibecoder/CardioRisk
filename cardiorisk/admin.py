from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user

from . import audit
from .extensions import db
from .models import ROLES, AuditEvent, User, temporary_password
from .tenancy import admin_required, get_scoped_or_404, scoped

bp = Blueprint("admin", __name__, url_prefix="/admin")
AUDIT_PER_PAGE = 50


def _other_active_admins(user: User) -> int:
    return scoped(User).filter(User.role == "admin", User.active.is_(True), User.id != user.id).count()


@bp.route("/users")
@admin_required
def users():
    return render_template("admin/users.html",
                           users=scoped(User).order_by(User.name).all(), roles=ROLES)


@bp.route("/users", methods=["POST"])
@admin_required
def create_user():
    email = (request.form.get("email") or "").strip().lower()
    name = (request.form.get("name") or "").strip()
    role = request.form.get("role")
    error = None
    if not name or len(name) > 200:
        error = "Enter a name (max 200 characters)."
    elif "@" not in email or len(email) > 254:
        error = "Enter a valid email address."
    elif role not in ROLES:
        error = "Choose a role."
    elif db.session.execute(db.select(User.id).filter_by(email=email)).first():
        # Emails are unique across all clinics; do not say which clinic has it.
        error = "That email address cannot be used."
    if error:
        return render_template("admin/users.html", users=scoped(User).order_by(User.name).all(),
                               roles=ROLES, error=error, form=request.form), 422
    password = temporary_password()
    user = User(org_id=current_user.org_id, email=email, name=name, role=role,
                must_change_password=True)
    user.set_password(password)
    db.session.add(user)
    db.session.flush()
    audit.record("user_created", user, role=role)
    db.session.commit()
    return render_template("admin/temp_password.html", user=user, password=password)


@bp.route("/users/<int:user_id>/<action>", methods=["POST"])
@admin_required
def user_action(user_id, action):
    user = get_scoped_or_404(User, user_id)
    is_self = user.id == current_user.id
    if action in ("deactivate", "make_clinician") and (is_self or (user.is_admin and not _other_active_admins(user))):
        flash("You cannot remove the last active administrator or your own access.", "error")
        return redirect(url_for("admin.users"))

    if action == "deactivate":
        user.active = False
        user.session_version += 1
    elif action == "activate":
        user.active = True
        user.locked_until = None
        user.failed_logins = 0
    elif action == "make_admin":
        user.role = "admin"
    elif action == "make_clinician":
        user.role = "clinician"
    elif action == "reset_mfa":
        user.mfa_secret = None
        user.mfa_last_counter = None
        user.session_version += 1
    elif action == "reset_password":
        password = temporary_password()
        user.set_password(password)
        user.must_change_password = True
        user.locked_until = None
        audit.record("user_reset_password", user)
        db.session.commit()
        return render_template("admin/temp_password.html", user=user, password=password)
    else:
        return redirect(url_for("admin.users"))
    audit.record(f"user_{action}", user)
    db.session.commit()
    flash(f"Updated {user.name}.")
    return redirect(url_for("admin.users"))


@bp.route("/settings", methods=["GET", "POST"])
@admin_required
def settings():
    org = current_user.organization
    if request.method == "POST":
        org.require_mfa = request.form.get("require_mfa") == "1"
        audit.record("org_settings_changed", org, require_mfa=org.require_mfa)
        db.session.commit()
        flash("Settings saved.")
        return redirect(url_for("admin.settings"))
    return render_template("admin/settings.html", org=org)


@bp.route("/audit")
@admin_required
def audit_log():
    query = scoped(AuditEvent).order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc())
    page = query.paginate(per_page=AUDIT_PER_PAGE, error_out=False)
    return render_template("admin/audit.html", page=page)
