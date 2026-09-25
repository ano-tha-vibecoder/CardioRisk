import click
from flask import Flask

from . import audit
from .extensions import db
from .models import Organization, User, temporary_password


def init_app(app: Flask) -> None:
    @app.cli.command("create-org")
    @click.argument("name")
    @click.option("--admin-email", required=True)
    @click.option("--admin-name", required=True)
    def create_org(name, admin_email, admin_name):
        """Onboard a clinic and its first administrator (platform operators only)."""
        email = admin_email.strip().lower()
        if db.session.execute(db.select(User.id).filter_by(email=email)).first():
            raise click.ClickException("a user with that email already exists")
        org = Organization(name=name.strip())
        db.session.add(org)
        db.session.flush()
        password = temporary_password()
        admin = User(org_id=org.id, email=email, name=admin_name.strip(), role="admin",
                     must_change_password=True)
        admin.set_password(password)
        db.session.add(admin)
        db.session.flush()
        audit.record("org_created", org, user=admin, via="cli")
        db.session.commit()
        click.echo(f"Created organisation {org.id} '{org.name}'.")
        click.echo(f"Admin {email} temporary password (shown once): {password}")
