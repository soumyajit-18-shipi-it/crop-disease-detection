"""One-time, idempotent migration of genuine user-owned SQLite records.

DATABASE_URL must contain the target PostgreSQL connection string. The script
never emits the URL or record contents, and deliberately excludes sessions,
OAuth states, unowned legacy rows, and obvious test identities.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def _timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        text = str(value or "").strip().replace("Z", "+00:00")
        result = datetime.fromisoformat(text) if text else datetime.now(timezone.utc)
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result


def _warnings(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _is_genuine_user(row: sqlite3.Row) -> bool:
    email = str(row["email"] or "").lower()
    provider_id = str(row["provider_account_id"] or "").lower()
    try:
        UUID(str(row["id"]))
    except ValueError:
        return False
    return (
        row["auth_provider"] == "google"
        and "@" in email
        and not email.endswith((".test", "@example.com", "@example.test"))
        and provider_id
        and not provider_id.startswith(("test", "google-test"))
    )


def migrate(source: Path, database_url: str) -> dict[str, int]:
    if not source.is_file():
        raise FileNotFoundError(f"SQLite source does not exist: {source}")
    if not database_url.startswith(("postgresql://", "postgres://")):
        raise RuntimeError("DATABASE_URL must be a PostgreSQL connection string.")

    with sqlite3.connect(source) as sqlite_connection:
        sqlite_connection.row_factory = sqlite3.Row
        users = [
            row
            for row in sqlite_connection.execute(
                "SELECT id, name, email, profile_picture, auth_provider, "
                "provider_account_id, created_at, last_login_at FROM users"
            )
            if _is_genuine_user(row)
        ]
        user_ids = {str(row["id"]) for row in users}
        diseases = list(sqlite_connection.execute("SELECT * FROM diseases"))
        scans = [
            row
            for row in sqlite_connection.execute("SELECT * FROM scans WHERE user_id IS NOT NULL")
            if str(row["user_id"]) in user_ids
        ]
        feedback = [
            row
            for row in sqlite_connection.execute("SELECT * FROM feedback WHERE user_id IS NOT NULL")
            if str(row["user_id"]) in user_ids
        ]

    with psycopg.connect(database_url, sslmode="require", row_factory=dict_row) as connection:
        with connection.transaction():
            for row in diseases:
                connection.execute(
                    """
                    INSERT INTO public.diseases(
                        class_name, crop, disease_name, symptoms,
                        recommended_treatment, severity_level
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT(class_name) DO UPDATE SET
                        crop = excluded.crop,
                        disease_name = excluded.disease_name,
                        symptoms = excluded.symptoms,
                        recommended_treatment = excluded.recommended_treatment,
                        severity_level = excluded.severity_level
                    """,
                    tuple(row),
                )
            for row in users:
                connection.execute(
                    """
                    INSERT INTO public.users(
                        id, name, email, profile_picture, auth_provider,
                        provider_account_id, created_at, last_login_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        email = excluded.email,
                        profile_picture = excluded.profile_picture,
                        last_login_at = excluded.last_login_at
                    """,
                    (
                        row["id"], row["name"], row["email"], row["profile_picture"],
                        row["auth_provider"], row["provider_account_id"],
                        _timestamp(row["created_at"]), _timestamp(row["last_login_at"]),
                    ),
                )
            for row in scans:
                connection.execute(
                    """
                    INSERT INTO public.scans(
                        id, user_id, timestamp, predicted_class, confidence, image_hash,
                        original_filename, content_type, file_size, model_name, model_version,
                        detection_status, quality_status, quality_warnings
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (
                        row["id"], row["user_id"], _timestamp(row["timestamp"]),
                        row["predicted_class"], row["confidence"], row["image_hash"],
                        row["original_filename"], row["content_type"], row["file_size"],
                        row["model_name"], row["model_version"], row["detection_status"],
                        row["quality_status"], Jsonb(_warnings(row["quality_warnings"])),
                    ),
                )
            scan_ids = {int(row["id"]) for row in scans}
            for row in feedback:
                scan_id = int(row["scan_id"]) if row["scan_id"] is not None else None
                if scan_id is not None and scan_id not in scan_ids:
                    continue
                connection.execute(
                    """
                    INSERT INTO public.feedback(
                        id, user_id, scan_id, timestamp, predicted_class, confidence, message
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (
                        row["id"], row["user_id"], scan_id, _timestamp(row["timestamp"]),
                        row["predicted_class"], row["confidence"], row["message"],
                    ),
                )
            for table in ("auth_sessions", "scans", "feedback"):
                connection.execute(
                    "SELECT setval(pg_get_serial_sequence(%s, 'id'), "
                    "GREATEST(COALESCE((SELECT max(id) FROM " + table + "), 0), 1), "
                    "EXISTS (SELECT 1 FROM " + table + "))",
                    (f"public.{table}",),
                )

        counts = {
            "users": connection.execute(
                "SELECT count(*) AS count FROM public.users WHERE id = ANY(%s::uuid[])",
                ([row["id"] for row in users],),
            ).fetchone()["count"],
            "scans": connection.execute(
                "SELECT count(*) AS count FROM public.scans WHERE user_id = ANY(%s::uuid[])",
                ([row["id"] for row in users],),
            ).fetchone()["count"],
            "feedback": connection.execute(
                "SELECT count(*) AS count FROM public.feedback WHERE user_id = ANY(%s::uuid[])",
                ([row["id"] for row in users],),
            ).fetchone()["count"],
        }
        orphan_counts = connection.execute(
            """
            SELECT
              (SELECT count(*) FROM public.scans s LEFT JOIN public.users u ON u.id=s.user_id WHERE u.id IS NULL)
                AS scans,
              (SELECT count(*) FROM public.feedback f LEFT JOIN public.users u ON u.id=f.user_id WHERE u.id IS NULL)
                AS feedback
            """
        ).fetchone()
    if orphan_counts["scans"] or orphan_counts["feedback"]:
        raise RuntimeError("Foreign-key verification found orphaned migrated records.")
    return {key: int(value) for key, value in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    database_url = os.getenv("DATABASE_URL", "").strip()
    counts = migrate(args.source.resolve(), database_url)
    print(
        "Migration verified: "
        f"users={counts['users']} scans={counts['scans']} feedback={counts['feedback']}"
    )


if __name__ == "__main__":
    main()
