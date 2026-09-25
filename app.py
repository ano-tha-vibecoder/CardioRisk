"""WSGI entry point: ``gunicorn app:app`` / ``flask --app app ...``."""
from cardiorisk import create_app

app = create_app()
