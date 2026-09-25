"""CardioRisk: multi-clinic coronary artery disease risk assessment."""
import logging
import os

from flask import Flask, render_template
from flask_wtf.csrf import CSRFError
from werkzeug.middleware.proxy_fix import ProxyFix

from . import config as app_config
from .config import ROOT
from .extensions import csrf, db, login_manager, migrate

log = logging.getLogger("cardiorisk")


def create_app(overrides: dict | None = None) -> Flask:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    app = Flask(__name__, template_folder=str(ROOT / "templates"),
                static_folder=str(ROOT / "static"), instance_path=str(ROOT / "instance"))
    production = os.environ.get("APP_ENV") == "production"
    if production:
        # Behind a TLS-terminating proxy (Heroku, Render, a load balancer):
        # trust one hop so request.is_secure and remote_addr are correct.
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    app.config.update(app_config.load(production, log))
    app.config.update(overrides or {})

    db.init_app(app)
    migrate.init_app(app, db, directory=str(ROOT / "migrations"))
    csrf.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please sign in to continue."
    login_manager.session_protection = "strong"

    from . import admin, auth, cli, main, models, patients, security, timezones  # noqa: F401

    @app.context_processor
    def globals_():
        from . import ml
        return {"model_metadata": ml.MODEL_METADATA, "now": models.utcnow()}

    timezones.init_app(app)
    security.init_app(app)
    cli.init_app(app)
    app.register_blueprint(main.bp)
    app.register_blueprint(auth.bp)
    app.register_blueprint(patients.bp)
    app.register_blueprint(admin.bp)
    _register_errors(app)
    return app


def _register_errors(app: Flask) -> None:
    def page(code, message):
        return render_template("error.html", code=code, message=message), code

    @app.errorhandler(CSRFError)
    def csrf_error(_):
        return page(400, "Your session expired. Go back, reload the page and try again.")

    @app.errorhandler(403)
    def forbidden(_):
        return page(403, "You do not have access to this page.")

    @app.errorhandler(404)
    def not_found(_):
        return page(404, "Page not found.")

    @app.errorhandler(413)
    def too_large(_):
        return page(413, "Request too large.")

    @app.errorhandler(500)
    def server_error(_):
        return page(500, "Something went wrong. The error has been logged.")
