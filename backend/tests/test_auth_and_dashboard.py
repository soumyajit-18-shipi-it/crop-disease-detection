from __future__ import annotations

from io import BytesIO
from urllib.parse import parse_qs, urlparse

import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from backend.api.auth import (
    CSRF_COOKIE,
    OAUTH_STATE_COOKIE,
    SESSION_COOKIE,
    create_session,
    hash_token,
    isoformat,
    utc_now,
)
from backend.config import PROJECT_ROOT, Settings, settings
from backend.db.database import connect_database
from backend.main import app


def textured_image_bytes(size: tuple[int, int] = (160, 160)) -> bytes:
    y, x = np.indices((size[1], size[0]))
    array = np.stack(((x * 3) % 255, (y * 5) % 255, ((x + y) * 2) % 255), axis=-1).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG")
    return buffer.getvalue()


def mock_google_profile(monkeypatch) -> None:
    from backend.api.routes import auth as auth_route

    async def google_profile(_code: str) -> dict:
        return {
            "sub": "google-user-123",
            "name": "Real Test User",
            "email": "real.user@example.test",
            "email_verified": True,
            "picture": "https://example.test/profile.jpg",
        }

    monkeypatch.setattr(auth_route, "_fetch_google_user", google_profile)


def complete_google_login(test_client: TestClient) -> tuple[str, str, str]:
    login = test_client.get("/auth/google/login", follow_redirects=False)
    assert login.status_code == 302
    state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
    callback = test_client.get(
        "/auth/google/callback",
        params={"code": "single-use-code", "state": state},
        follow_redirects=False,
    )
    assert callback.status_code == 302
    assert callback.headers["location"] == "http://127.0.0.1:5173/?auth=success"
    session_token = test_client.cookies.get(SESSION_COOKIE)
    csrf_token = test_client.cookies.get(CSRF_COOKIE)
    assert session_token
    assert csrf_token
    test_client.headers["X-CSRF-Token"] = csrf_token
    return state, session_token, csrf_token


def test_auth_configuration_reports_missing_credentials(unauthenticated_client, monkeypatch):
    monkeypatch.setattr(settings, "google_client_id", "")
    response = unauthenticated_client.get("/auth/config")
    assert response.status_code == 200
    assert response.json() == {"provider": "google", "configured": False, "callback_url": None}
    login = unauthenticated_client.get("/auth/google/login", follow_redirects=False)
    assert login.status_code == 503
    assert "GOOGLE_CLIENT_ID" in login.json()["detail"]


def test_google_login_uses_exact_callback_and_one_time_state(unauthenticated_client):
    response = unauthenticated_client.get("/auth/google/login", follow_redirects=False)
    assert response.status_code == 302
    parsed = urlparse(response.headers["location"])
    query = parse_qs(parsed.query)
    assert parsed.netloc == "accounts.google.com"
    assert query["redirect_uri"] == [settings.oauth_callback_url]
    assert query["scope"] == ["openid email profile"]
    assert query["response_type"] == ["code"]
    assert query["prompt"] == ["select_account"]
    assert query["state"][0]
    assert unauthenticated_client.cookies.get("leaflight_oauth_state") == query["state"][0]
    with connect_database() as connection:
        assert connection.execute("SELECT COUNT(*) FROM oauth_states").fetchone()[0] == 1


def test_google_callback_creates_user_and_session_without_storing_tokens(unauthenticated_client, monkeypatch):
    mock_google_profile(monkeypatch)
    login = unauthenticated_client.get("/auth/google/login", follow_redirects=False)
    state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
    callback = unauthenticated_client.get(
        "/auth/google/callback",
        params={"code": "single-use-code", "state": state},
        follow_redirects=False,
    )
    assert callback.status_code == 302
    assert callback.headers["location"] == "http://127.0.0.1:5173/?auth=success"
    assert unauthenticated_client.cookies.get(SESSION_COOKIE)
    assert unauthenticated_client.cookies.get(CSRF_COOKIE)
    session = unauthenticated_client.get("/auth/session")
    assert session.status_code == 200
    assert session.json()["user"]["email"] == "real.user@example.test"
    with connect_database() as connection:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(auth_sessions)")}
        assert "access_token" not in columns
        assert "refresh_token" not in columns
        assert connection.execute("SELECT COUNT(*) FROM oauth_states").fetchone()[0] == 0


def test_invalid_oauth_state_is_rejected(unauthenticated_client):
    response = unauthenticated_client.get(
        "/auth/google/callback",
        params={"code": "code", "state": "attacker-state"},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert response.headers["location"] == "http://127.0.0.1:5173/?auth_error=state"
    assert unauthenticated_client.cookies.get(OAUTH_STATE_COOKIE) is None


def test_cancelled_google_login_returns_to_login_screen(unauthenticated_client):
    login = unauthenticated_client.get("/auth/google/login", follow_redirects=False)
    state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
    response = unauthenticated_client.get(
        "/auth/google/callback",
        params={"error": "access_denied", "state": state},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert response.headers["location"] == "http://127.0.0.1:5173/?auth_error=cancelled"


def test_duplicate_email_with_different_google_subject_is_rejected(client):
    from backend.api.routes.auth import _upsert_google_user

    with pytest.raises(HTTPException) as error:
        _upsert_google_user(
            {
                "sub": "unexpected-google-subject",
                "name": "Different Subject",
                "email": "farmer@example.test",
                "email_verified": True,
            }
        )
    assert getattr(error.value, "status_code", None) == 409


def test_logout_revokes_the_server_session(client):
    response = client.post("/auth/logout")
    assert response.status_code == 204
    assert client.get("/auth/session").status_code == 401


def test_each_login_uses_fresh_state_and_replaces_stale_state(unauthenticated_client):
    first = unauthenticated_client.get("/auth/google/login", follow_redirects=False)
    first_query = parse_qs(urlparse(first.headers["location"]).query)
    first_state = first_query["state"][0]

    second = unauthenticated_client.get("/auth/google/login", follow_redirects=False)
    second_query = parse_qs(urlparse(second.headers["location"]).query)
    second_state = second_query["state"][0]

    assert first_state != second_state
    assert first_query["redirect_uri"] == second_query["redirect_uri"] == [
        "http://127.0.0.1:8000/auth/google/callback"
    ]
    assert first_query["prompt"] == second_query["prompt"] == ["select_account"]
    assert unauthenticated_client.cookies.get(OAUTH_STATE_COOKIE) == second_state
    with connect_database() as connection:
        assert connection.execute("SELECT COUNT(*) FROM oauth_states").fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM oauth_states WHERE state_hash = ?",
            (hash_token(first_state),),
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM oauth_states WHERE state_hash = ?",
            (hash_token(second_state),),
        ).fetchone()[0] == 1


def test_login_logout_login_creates_a_new_working_session(unauthenticated_client, monkeypatch):
    mock_google_profile(monkeypatch)
    _, first_session, first_csrf = complete_google_login(unauthenticated_client)
    assert unauthenticated_client.get("/dashboard").status_code == 200

    logout_response = unauthenticated_client.post("/auth/logout")
    assert logout_response.status_code == 204
    assert unauthenticated_client.cookies.get(SESSION_COOKIE) is None
    assert unauthenticated_client.cookies.get(CSRF_COOKIE) is None
    assert unauthenticated_client.get("/auth/session").status_code == 401
    assert unauthenticated_client.get("/dashboard").status_code == 401

    _, second_session, second_csrf = complete_google_login(unauthenticated_client)
    assert second_session != first_session
    assert second_csrf != first_csrf
    assert unauthenticated_client.get("/auth/session").status_code == 200
    assert unauthenticated_client.get("/dashboard").status_code == 200
    with connect_database() as connection:
        rows = connection.execute(
            "SELECT token_hash, revoked_at FROM auth_sessions ORDER BY id"
        ).fetchall()
    assert len(rows) == 2
    assert rows[0]["token_hash"] == hash_token(first_session)
    assert rows[0]["revoked_at"] is not None
    assert rows[1]["token_hash"] == hash_token(second_session)
    assert rows[1]["revoked_at"] is None


def test_login_logout_login_logout_login_succeeds(unauthenticated_client, monkeypatch):
    mock_google_profile(monkeypatch)
    session_tokens: list[str] = []
    oauth_states: list[str] = []
    for attempt in range(3):
        oauth_state, session_token, _ = complete_google_login(unauthenticated_client)
        oauth_states.append(oauth_state)
        session_tokens.append(session_token)
        assert unauthenticated_client.get("/dashboard").status_code == 200
        if attempt < 2:
            assert unauthenticated_client.post("/auth/logout").status_code == 204
            assert unauthenticated_client.get("/dashboard").status_code == 401

    assert len(set(oauth_states)) == 3
    assert len(set(session_tokens)) == 3
    with connect_database() as connection:
        rows = connection.execute(
            "SELECT revoked_at FROM auth_sessions ORDER BY id"
        ).fetchall()
    assert [row["revoked_at"] is not None for row in rows] == [True, True, False]


def test_consumed_oauth_state_cannot_be_reused(unauthenticated_client, monkeypatch):
    mock_google_profile(monkeypatch)
    complete_google_login(unauthenticated_client)
    replay_client = TestClient(app)
    replay_login = replay_client.get("/auth/google/login", follow_redirects=False)
    consumed_state = parse_qs(urlparse(replay_login.headers["location"]).query)["state"][0]
    with connect_database() as connection:
        connection.execute(
            "DELETE FROM oauth_states WHERE state_hash = ?",
            (hash_token(consumed_state),),
        )
        connection.commit()

    replay = replay_client.get(
        "/auth/google/callback",
        params={"code": "replayed-code", "state": consumed_state},
        follow_redirects=False,
    )

    assert replay.status_code == 302
    assert replay.headers["location"] == "http://127.0.0.1:5173/?auth_error=state"
    assert replay_client.cookies.get(OAUTH_STATE_COOKIE) is None
    with connect_database() as connection:
        assert connection.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 1


def test_expired_state_is_rejected_without_blocking_new_login(unauthenticated_client, monkeypatch):
    mock_google_profile(monkeypatch)
    login = unauthenticated_client.get("/auth/google/login", follow_redirects=False)
    expired_state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
    with connect_database() as connection:
        connection.execute(
            "UPDATE oauth_states SET expires_at = '2000-01-01T00:00:00+00:00' "
            "WHERE state_hash = ?",
            (hash_token(expired_state),),
        )
        connection.commit()

    expired = unauthenticated_client.get(
        "/auth/google/callback",
        params={"code": "expired-code", "state": expired_state},
        follow_redirects=False,
    )
    assert expired.status_code == 302
    assert expired.headers["location"] == "http://127.0.0.1:5173/?auth_error=state"
    assert unauthenticated_client.cookies.get(OAUTH_STATE_COOKIE) is None
    with connect_database() as connection:
        assert connection.execute("SELECT COUNT(*) FROM oauth_states").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 0

    fresh_state, fresh_session, _ = complete_google_login(unauthenticated_client)
    assert fresh_state != expired_state
    assert fresh_session
    assert unauthenticated_client.get("/dashboard").status_code == 200


def test_logout_clears_all_auth_cookies_with_matching_attributes(unauthenticated_client, monkeypatch):
    mock_google_profile(monkeypatch)
    complete_google_login(unauthenticated_client)
    pending_login = unauthenticated_client.get("/auth/google/login", follow_redirects=False)
    pending_cookie = next(
        header
        for header in pending_login.headers.get_list("set-cookie")
        if header.startswith(f"{OAUTH_STATE_COOKIE}=")
    ).lower()
    assert "path=/" in pending_cookie
    assert "samesite=lax" in pending_cookie
    assert "httponly" in pending_cookie
    assert "secure" not in pending_cookie
    assert "domain=" not in pending_cookie

    logout_response = unauthenticated_client.post("/auth/logout")
    assert logout_response.status_code == 204
    deletion_headers = {
        header.split("=", 1)[0]: header.lower()
        for header in logout_response.headers.get_list("set-cookie")
    }
    assert set(deletion_headers) == {SESSION_COOKIE, CSRF_COOKIE, OAUTH_STATE_COOKIE}
    for header in deletion_headers.values():
        assert "path=/" in header
        assert "samesite=lax" in header
        assert "max-age=0" in header
        assert "secure" not in header
        assert "domain=" not in header
    assert "httponly" in deletion_headers[SESSION_COOKIE]
    assert "httponly" not in deletion_headers[CSRF_COOKIE]
    assert "httponly" in deletion_headers[OAUTH_STATE_COOKIE]
    assert unauthenticated_client.cookies.get(SESSION_COOKIE) is None
    assert unauthenticated_client.cookies.get(CSRF_COOKIE) is None
    assert unauthenticated_client.cookies.get(OAUTH_STATE_COOKIE) is None
    with connect_database() as connection:
        assert connection.execute("SELECT COUNT(*) FROM oauth_states").fetchone()[0] == 0


def test_public_auth_routes_remain_accessible_after_logout(unauthenticated_client, monkeypatch):
    mock_google_profile(monkeypatch)
    complete_google_login(unauthenticated_client)
    assert unauthenticated_client.post("/auth/logout").status_code == 204
    assert unauthenticated_client.get("/dashboard").status_code == 401

    assert unauthenticated_client.get("/health").status_code != 401
    assert unauthenticated_client.get("/auth/config").status_code == 200
    assert unauthenticated_client.get("/auth/google/login", follow_redirects=False).status_code == 302
    callback = unauthenticated_client.get(
        "/auth/google/callback",
        params={"state": "not-the-current-state"},
        follow_redirects=False,
    )
    assert callback.status_code == 302
    assert "auth_error=state" in callback.headers["location"]


def test_local_auth_configuration_uses_only_127_origin(monkeypatch):
    monkeypatch.setenv("APP_URL", "http://127.0.0.1:5173")
    monkeypatch.setenv(
        "OAUTH_CALLBACK_URL",
        "http://127.0.0.1:8000/auth/google/callback",
    )
    monkeypatch.setenv("CORS_ORIGINS", "http://127.0.0.1:5173")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("COOKIE_SAMESITE", "lax")
    monkeypatch.setenv("COOKIE_DOMAIN", "")
    local_settings = Settings()
    assert local_settings.app_url == "http://127.0.0.1:5173"
    assert local_settings.oauth_callback_url == "http://127.0.0.1:8000/auth/google/callback"
    assert local_settings.cors_origins == ["http://127.0.0.1:5173"]
    assert local_settings.cookie_secure is False
    assert local_settings.cookie_samesite == "lax"
    assert local_settings.cookie_domain is None

    frontend_example = (PROJECT_ROOT / "frontend" / ".env.example").read_text(encoding="utf-8")
    assert "VITE_API_URL=http://127.0.0.1:8000" in frontend_example
    assert "localhost" not in frontend_example


def test_protected_routes_require_authentication(unauthenticated_client):
    assert unauthenticated_client.get("/dashboard").status_code == 401
    assert unauthenticated_client.get("/history").status_code == 401
    assert unauthenticated_client.post(
        "/predict", files={"file": ("leaf.jpg", textured_image_bytes(), "image/jpeg")}
    ).status_code == 401


def test_mutating_routes_require_csrf(client):
    client.headers.pop("X-CSRF-Token")
    response = client.post(
        "/predict", files={"file": ("leaf.jpg", textured_image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 403


def test_dashboard_uses_only_authenticated_users_records(client):
    now = isoformat(utc_now())
    with connect_database() as connection:
        connection.execute(
            "INSERT INTO scans(user_id, predicted_class, confidence, image_hash, detection_status) VALUES (?, ?, ?, ?, ?)",
            ("test-user", "Tomato_healthy", 0.91, "a" * 64, "healthy"),
        )
        connection.execute(
            "INSERT INTO scans(user_id, predicted_class, confidence, image_hash, detection_status) VALUES (?, ?, ?, ?, ?)",
            ("test-user", "Tomato_Late_blight", 0.82, "b" * 64, "disease_detected"),
        )
        connection.execute(
            """
            INSERT INTO users(id, name, email, auth_provider, provider_account_id, created_at, last_login_at)
            VALUES ('other-user', 'Other', 'other@example.test', 'google', 'other-google', ?, ?)
            """,
            (now, now),
        )
        connection.execute(
            "INSERT INTO scans(user_id, predicted_class, confidence, image_hash, detection_status) VALUES ('other-user', 'Potato___Late_blight', .99, ?, 'disease_detected')",
            ("c" * 64,),
        )
        connection.commit()
    payload = client.get("/dashboard").json()
    assert payload["total_scans"] == 2
    assert payload["healthy_scans"] == 1
    assert payload["diseased_scans"] == 1
    assert payload["average_confidence"] == 0.865
    assert payload["disease_distribution"][0]["class_name"] == "Tomato_Late_blight"
    history = client.get("/history").json()
    assert len(history) == 2
    assert all(item["predicted_class"] != "Potato___Late_blight" for item in history)


def test_expired_session_is_rejected(isolated_database):
    now = isoformat(utc_now())
    with connect_database() as connection:
        connection.execute(
            """
            INSERT INTO users(id, name, email, auth_provider, provider_account_id, created_at, last_login_at)
            VALUES ('expired-user', 'Expired', 'expired@example.test', 'google', 'expired-google', ?, ?)
            """,
            (now, now),
        )
        connection.commit()
    token, csrf, _ = create_session("expired-user")
    with connect_database() as connection:
        connection.execute("UPDATE auth_sessions SET expires_at = '2000-01-01T00:00:00+00:00'")
        connection.commit()
    expired_client = TestClient(app)
    expired_client.cookies.set(SESSION_COOKIE, token)
    expired_client.cookies.set(CSRF_COOKIE, csrf)
    assert expired_client.get("/auth/session").status_code == 401


def test_tiny_and_contentless_images_are_rejected(client):
    tiny = BytesIO()
    Image.new("RGB", (32, 32), "green").save(tiny, format="PNG")
    assert client.post("/predict", files={"file": ("tiny.png", tiny.getvalue(), "image/png")}).status_code == 422
    blank = BytesIO()
    Image.new("RGB", (128, 128), "green").save(blank, format="PNG")
    response = client.post("/predict", files={"file": ("blank.png", blank.getvalue(), "image/png")})
    assert response.status_code == 422
    assert "visual detail" in response.json()["detail"]
