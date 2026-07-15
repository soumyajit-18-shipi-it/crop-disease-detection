from __future__ import annotations

import os
from uuid import uuid4

import pytest

from backend.api.auth import create_session, hash_token, isoformat, utc_now
from backend.config import PROJECT_ROOT, settings
from backend.db.database import (
    DatabaseConfigurationError,
    close_database_pool,
    connect_database,
    validate_database_configuration,
)


def test_production_refuses_sqlite(monkeypatch):
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "database_url", "sqlite:///backend/db/unsafe.db")
    with pytest.raises(DatabaseConfigurationError, match="Production requires"):
        validate_database_configuration()


def test_postgres_migration_contains_portable_production_schema():
    migration = (
        PROJECT_ROOT / "supabase" / "migrations" / "20260715170000_initial_production_schema.sql"
    ).read_text(encoding="utf-8")
    uppercase = migration.upper()
    assert "AUTOINCREMENT" not in uppercase
    assert "PRAGMA" not in uppercase
    assert "INSERT OR" not in uppercase
    assert "TIMESTAMP WITH TIME ZONE" in uppercase
    assert "JSONB" in uppercase
    assert "FOREIGN KEY (SCAN_ID, USER_ID)" in uppercase
    assert "ENABLE ROW LEVEL SECURITY" in uppercase


@pytest.mark.postgres
def test_postgres_session_state_and_prediction_persistence(monkeypatch):
    database_url = os.getenv("TEST_DATABASE_URL", "").strip()
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")

    user_id = str(uuid4())
    provider_id = f"integration-{uuid4()}"
    state_value = f"state-{uuid4()}"
    monkeypatch.setattr(settings, "environment", "test")
    monkeypatch.setattr(settings, "database_url", database_url)
    close_database_pool()
    try:
        now = isoformat(utc_now())
        with connect_database() as connection:
            connection.execute(
                """
                INSERT INTO users(
                    id, name, email, auth_provider, provider_account_id,
                    created_at, last_login_at
                ) VALUES (?, ?, ?, 'google', ?, ?, ?)
                """,
                (
                    user_id,
                    "PostgreSQL Integration",
                    f"{user_id}@example.test",
                    provider_id,
                    now,
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO oauth_states(state_hash, return_to, expires_at) "
                "VALUES (?, '/', ?)",
                (hash_token(state_value), "2999-01-01T00:00:00+00:00"),
            )
            connection.commit()

        with connect_database() as connection:
            consumed = connection.execute(
                "DELETE FROM oauth_states WHERE state_hash = ? RETURNING return_to",
                (hash_token(state_value),),
            ).fetchone()
            replay = connection.execute(
                "DELETE FROM oauth_states WHERE state_hash = ? RETURNING return_to",
                (hash_token(state_value),),
            ).fetchone()
            connection.commit()
        assert consumed["return_to"] == "/"
        assert replay is None

        session_token, _, _ = create_session(user_id)
        with connect_database() as connection:
            session = connection.execute(
                "SELECT id FROM auth_sessions WHERE token_hash = ? AND revoked_at IS NULL",
                (hash_token(session_token),),
            ).fetchone()
            scan = connection.execute(
                """
                INSERT INTO scans(
                    user_id, predicted_class, confidence, image_hash,
                    model_name, model_version, detection_status,
                    quality_status, quality_warnings
                ) VALUES (?, 'Tomato_healthy', 0.9, ?, 'integration', 'test',
                          'healthy', 'acceptable', ?)
                RETURNING id
                """,
                (user_id, "0" * 64, "[]"),
            ).fetchone()
            connection.commit()
        assert session is not None
        assert int(scan["id"]) > 0
    finally:
        try:
            with connect_database() as connection:
                connection.execute("DELETE FROM users WHERE id = ?", (user_id,))
                connection.commit()
        finally:
            close_database_pool()
