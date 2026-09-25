import pyotp

from cardiorisk.extensions import db
from cardiorisk.models import AuditEvent, User
from tests.conftest import PASSWORD, csrf_token, login, make_org, make_user


def _actions():
    return [e.action for e in db.session.query(AuditEvent).order_by(AuditEvent.id)]


def test_protected_pages_redirect_to_login(client):
    for path in ("/dashboard", "/patients", "/patients/1", "/assessments/1", "/admin/users"):
        response = client.get(path)
        assert response.status_code == 302 and "/login" in response.headers["Location"], path


def test_login_success_and_logout(app, client):
    make_user(make_org())
    response = login(client)
    assert response.status_code == 302 and response.headers["Location"].endswith("/dashboard")
    assert client.get("/dashboard").status_code == 200
    assert client.get("/logout").status_code == 405  # logout is POST-only (CSRF-protected)
    client.post("/logout")
    assert client.get("/dashboard").status_code == 302
    assert _actions() == ["login", "logout"]


def test_failed_login_messages_do_not_reveal_accounts(app, client):
    make_user(make_org())
    wrong = login(client, password="nope").get_data(as_text=True)
    unknown = login(client, email="ghost@a.test").get_data(as_text=True)
    assert "Sign-in failed" in wrong and "Sign-in failed" in unknown
    assert _actions() == ["login_failed", "login_failed"]


def test_lockout_after_repeated_failures(app, client):
    user = make_user(make_org())
    for _ in range(app.config["LOGIN_MAX_FAILURES"]):
        assert login(client, password="wrong").status_code == 401
    assert db.session.get(User, user.id).locked_until is not None
    assert login(client).status_code == 401  # correct password, still locked
    assert "account_locked" in _actions()


def test_deactivated_user_cannot_log_in(app, client):
    make_user(make_org(), active=False)
    assert login(client).status_code == 401


def test_forced_password_change(app, client):
    make_user(make_org(), must_change_password=True)
    login(client)
    response = client.get("/dashboard")
    assert response.status_code == 302 and response.headers["Location"].endswith("/account/password")
    bad = client.post("/account/password", data={"current": PASSWORD, "new": "short", "confirm": "short"})
    assert bad.status_code == 400
    new = "a much longer passphrase"
    ok = client.post("/account/password", data={"current": PASSWORD, "new": new, "confirm": new})
    assert ok.status_code == 302
    assert client.get("/dashboard").status_code == 200  # this session survives the change


def test_password_change_signs_out_other_sessions(app):
    make_user(make_org())
    first, second = app.test_client(), app.test_client()
    login(first)
    login(second)
    new = "a much longer passphrase"
    first.post("/account/password", data={"current": PASSWORD, "new": new, "confirm": new})
    assert first.get("/dashboard").status_code == 200
    assert second.get("/dashboard").status_code == 302


def test_mfa_enrolment_and_login(app, client):
    user = make_user(make_org())
    login(client)
    client.get("/account/mfa")
    with client.session_transaction() as s:
        secret = s["mfa_setup_secret"]
    assert client.post("/account/mfa", data={"code": "000000"}).status_code == 400
    assert client.post("/account/mfa", data={"code": pyotp.TOTP(secret).now()}).status_code == 302
    assert db.session.get(User, user.id).mfa_secret == secret
    client.post("/logout")

    response = login(client)
    assert response.headers["Location"].endswith("/login/mfa")
    assert client.get("/dashboard").status_code == 302  # password alone is not enough
    assert client.post("/login/mfa", data={"code": "123456"}).status_code == 401
    # The code used at enrolment is rejected (replay); wait for the next step is
    # impractical in a test, so move the stored counter back one step instead.
    stored = db.session.get(User, user.id)
    stored.mfa_last_counter -= 1
    db.session.commit()
    code = pyotp.TOTP(secret).now()
    assert client.post("/login/mfa", data={"code": code}).status_code == 302
    assert client.get("/dashboard").status_code == 200
    client.post("/logout")
    login(client)
    assert client.post("/login/mfa", data={"code": code}).status_code == 401  # same code again


def test_org_can_require_mfa(app, client):
    make_user(make_org(require_mfa=True))
    login(client)
    response = client.get("/patients")
    assert response.status_code == 302 and response.headers["Location"].endswith("/account/mfa")


def test_open_redirect_is_blocked(app, client):
    make_user(make_org())
    client.get("/login?next=https://evil.example/")
    assert login(client).headers["Location"].endswith("/dashboard")
    client.post("/logout")
    client.get("/login?next=//evil.example/")
    assert login(client).headers["Location"].endswith("/dashboard")
    client.post("/logout")
    client.get("/login?next=/patients")
    assert login(client).headers["Location"].endswith("/patients")


def test_absolute_session_timeout(app, client):
    make_user(make_org())
    login(client)
    app.config["SESSION_ABSOLUTE_HOURS"] = 0
    response = client.get("/dashboard")
    assert response.status_code == 302 and "expired=1" in response.headers["Location"]


def test_login_requires_csrf_token(csrf_app):
    with csrf_app.app_context():
        make_user(make_org())
    client = csrf_app.test_client()
    assert login(client).status_code == 400
    token = csrf_token(client.get("/login").get_data(as_text=True))
    response = client.post("/login", data={"email": "doc@a.test", "password": PASSWORD, "csrf_token": token})
    assert response.status_code == 302
