from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.auth import CSRF_COOKIE, SESSION_COOKIE, create_session, isoformat, utc_now
from backend.config import settings
from backend.db.seed_disease_data import seed_database
from backend.main import app


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    """Keep backend tests away from the repository database."""
    database_path = tmp_path / "test_disease_info.db"
    monkeypatch.setattr(settings, "environment", "test")
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{database_path.as_posix()}")
    monkeypatch.setattr(settings, "db_path", database_path)
    monkeypatch.setattr(settings, "auth_secret", "test-auth-secret-with-at-least-32-characters")
    monkeypatch.setattr(settings, "cookie_secure", False)
    monkeypatch.setattr(settings, "cookie_samesite", "lax")
    monkeypatch.setattr(settings, "cookie_domain", None)
    monkeypatch.setattr(settings, "app_url", "http://127.0.0.1:5173")
    monkeypatch.setattr(
        settings,
        "oauth_callback_url",
        "http://127.0.0.1:8000/auth/google/callback",
    )
    monkeypatch.setattr(settings, "cors_origins", ["http://127.0.0.1:5173"])
    monkeypatch.setattr(settings, "google_client_id", "test-client-id")
    monkeypatch.setattr(settings, "google_client_secret", "test-client-secret")
    seed_database(database_path)
    return database_path


@pytest.fixture
def client(isolated_database):
    """Create the API client only after the temporary DB path is active."""
    from backend.db.database import connect_database

    user_id = "test-user"
    now = isoformat(utc_now())
    with connect_database() as connection:
        connection.execute(
            """
            INSERT INTO users(
                id, name, email, auth_provider, provider_account_id,
                created_at, last_login_at
            ) VALUES (?, ?, ?, 'google', ?, ?, ?)
            """,
            (user_id, "Test Farmer", "farmer@example.test", "google-test-user", now, now),
        )
        connection.commit()
    session_token, csrf_token, _ = create_session(user_id)
    test_client = TestClient(app)
    test_client.cookies.set(SESSION_COOKIE, session_token)
    test_client.cookies.set(CSRF_COOKIE, csrf_token)
    test_client.headers["X-CSRF-Token"] = csrf_token
    return test_client


@pytest.fixture
def unauthenticated_client(isolated_database):
    return TestClient(app)
