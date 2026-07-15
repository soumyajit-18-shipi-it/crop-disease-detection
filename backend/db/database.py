from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import PROJECT_ROOT, settings


class DatabaseConfigurationError(RuntimeError):
    """The configured database is missing, unsafe, or unsupported."""


class ClosingConnection(sqlite3.Connection):
    """SQLite connection whose context manager also releases the file handle."""

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class HybridRow(dict[str, Any]):
    """Mapping row that also preserves SQLite-compatible numeric indexing."""

    def __init__(self, row: Mapping[str, Any]):
        super().__init__(row)
        self._values = tuple(row.values())

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return super().__getitem__(key)


def _adapt_row(row: Any) -> Any:
    if row is None or isinstance(row, sqlite3.Row):
        return row
    if isinstance(row, Mapping):
        return HybridRow(row)
    return row


class CursorAdapter:
    def __init__(self, cursor: Any):
        self._cursor = cursor

    @property
    def rowcount(self) -> int:
        return int(self._cursor.rowcount)

    @property
    def lastrowid(self) -> Any:
        return getattr(self._cursor, "lastrowid", None)

    @property
    def description(self) -> Any:
        return self._cursor.description

    def fetchone(self) -> Any:
        return _adapt_row(self._cursor.fetchone())

    def fetchall(self) -> list[Any]:
        return [_adapt_row(row) for row in self._cursor.fetchall()]

    def __iter__(self) -> Iterator[Any]:
        for row in self._cursor:
            yield _adapt_row(row)


def _postgres_query(sql: str) -> str:
    # Application SQL uses the SQLite qmark style. Psycopg uses %s, and any
    # literal percent signs in SQL must be escaped before parameters are bound.
    return sql.replace("%", "%%").replace("?", "%s")


class DatabaseConnection:
    def __init__(self, raw_connection: Any, dialect: str):
        self.raw_connection = raw_connection
        self.dialect = dialect

    def execute(self, sql: str, parameters: Sequence[Any] | None = None) -> CursorAdapter:
        bound = tuple(parameters or ())
        if self.dialect == "postgresql":
            cursor = self.raw_connection.cursor()
            cursor.execute(_postgres_query(sql), bound if bound else None)
            return CursorAdapter(cursor)
        return CursorAdapter(self.raw_connection.execute(sql, bound))

    def executescript(self, sql: str) -> None:
        if self.dialect != "sqlite":
            raise DatabaseConfigurationError("SQLite schema scripts cannot run on PostgreSQL.")
        self.raw_connection.executescript(sql)

    def commit(self) -> None:
        self.raw_connection.commit()

    def rollback(self) -> None:
        self.raw_connection.rollback()


_postgres_pool: Any | None = None
_postgres_pool_url: str | None = None
_pool_lock = threading.Lock()


def database_backend() -> str:
    database_url = settings.database_url.strip()
    if database_url.startswith(("postgresql://", "postgres://")):
        return "postgresql"
    if database_url.startswith("sqlite:///"):
        return "sqlite"
    if not database_url:
        raise DatabaseConfigurationError(
            "DATABASE_URL is required. Use a postgresql:// URL in production or an explicit "
            "sqlite:/// path for local development and tests."
        )
    raise DatabaseConfigurationError("DATABASE_URL must use postgresql:// or sqlite:///.")


def sqlite_path_from_url(database_url: str | None = None) -> Path | str:
    value = (database_url or settings.database_url).strip()
    if not value.startswith("sqlite:///"):
        raise DatabaseConfigurationError("The configured DATABASE_URL is not SQLite.")
    raw_path = value.removeprefix("sqlite:///")
    if raw_path == ":memory:":
        return ":memory:"
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_database_configuration() -> str:
    backend = database_backend()
    if settings.environment == "production" and backend != "postgresql":
        raise DatabaseConfigurationError(
            "Production requires a PostgreSQL DATABASE_URL; SQLite is not permitted."
        )
    if backend == "sqlite":
        target = sqlite_path_from_url()
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)
    return backend


def _get_postgres_pool() -> Any:
    global _postgres_pool, _postgres_pool_url
    database_url = settings.database_url.strip()
    with _pool_lock:
        if _postgres_pool is not None and _postgres_pool_url == database_url:
            return _postgres_pool
        if _postgres_pool is not None:
            _postgres_pool.close()
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise DatabaseConfigurationError(
                "PostgreSQL support requires psycopg and psycopg-pool."
            ) from exc
        _postgres_pool = ConnectionPool(
            conninfo=database_url,
            min_size=settings.database_pool_min_size,
            max_size=settings.database_pool_max_size,
            kwargs={
                "sslmode": "require",
                "row_factory": dict_row,
                "prepare_threshold": None,
            },
            open=False,
        )
        _postgres_pool.open(wait=True, timeout=settings.database_connect_timeout_seconds)
        _postgres_pool_url = database_url
        return _postgres_pool


def close_database_pool() -> None:
    global _postgres_pool, _postgres_pool_url
    with _pool_lock:
        if _postgres_pool is not None:
            _postgres_pool.close()
        _postgres_pool = None
        _postgres_pool_url = None


@contextmanager
def connect_database(path: str | Path | None = None) -> Iterator[DatabaseConnection]:
    """Open an explicit SQLite connection or a pooled production PostgreSQL connection."""
    if path is not None:
        backend = "sqlite"
        target: Path | str = Path(path).resolve()
    else:
        backend = database_backend()
        target = sqlite_path_from_url() if backend == "sqlite" else ""

    if backend == "sqlite":
        connection = sqlite3.connect(target, timeout=10, factory=ClosingConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield DatabaseConnection(connection, "sqlite")
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return

    pool = _get_postgres_pool()
    with pool.connection() as connection:
        yield DatabaseConnection(connection, "postgresql")


def database_json(value: Any) -> Any:
    if database_backend() == "postgresql":
        from psycopg.types.json import Jsonb

        return Jsonb(value)
    import json

    return json.dumps(value)


def database_error_types() -> tuple[type[BaseException], ...]:
    """Return driver exceptions without making psycopg a local-only dependency."""
    errors: list[type[BaseException]] = [sqlite3.Error]
    try:
        from psycopg import Error as PsycopgError
    except ImportError:
        return tuple(errors)
    errors.append(PsycopgError)
    return tuple(errors)


def timestamp_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        normalized = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return normalized.astimezone(timezone.utc).isoformat(timespec="seconds")
    return str(value)


def _columns(connection: DatabaseConnection, table: str) -> set[str]:
    return {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _add_column(connection: DatabaseConnection, table: str, definition: str) -> None:
    column_name = definition.split()[0]
    if column_name not in _columns(connection, table):
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


def migrate_database(path: str | Path | None = None) -> None:
    """Apply the legacy SQLite schema only for explicit local/test databases."""
    if path is None:
        if database_backend() != "sqlite":
            raise DatabaseConfigurationError(
                "PostgreSQL schema changes must be applied from supabase/migrations."
            )
        target = sqlite_path_from_url()
    else:
        target = Path(path).resolve()
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    with connect_database(target) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS diseases (
                class_name TEXT PRIMARY KEY,
                crop TEXT,
                disease_name TEXT,
                symptoms TEXT NOT NULL,
                recommended_treatment TEXT NOT NULL,
                severity_level TEXT
            );

            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                predicted_class TEXT NOT NULL,
                confidence REAL,
                message TEXT
            );

            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                profile_picture TEXT,
                auth_provider TEXT NOT NULL,
                provider_account_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_login_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(auth_provider, provider_account_id)
            );

            CREATE TABLE IF NOT EXISTS auth_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT NOT NULL UNIQUE,
                csrf_token_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT NOT NULL,
                revoked_at TEXT
            );

            CREATE TABLE IF NOT EXISTS oauth_states (
                state_hash TEXT PRIMARY KEY,
                return_to TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT NOT NULL
            );
            """
        )
        _add_column(connection, "scans", "user_id TEXT REFERENCES users(id) ON DELETE CASCADE")
        _add_column(connection, "scans", "original_filename TEXT")
        _add_column(connection, "scans", "content_type TEXT")
        _add_column(connection, "scans", "file_size INTEGER")
        _add_column(connection, "scans", "model_name TEXT")
        _add_column(connection, "scans", "model_version TEXT")
        _add_column(connection, "scans", "detection_status TEXT")
        _add_column(connection, "scans", "quality_status TEXT")
        _add_column(connection, "scans", "quality_warnings TEXT")
        _add_column(connection, "feedback", "user_id TEXT REFERENCES users(id) ON DELETE CASCADE")
        _add_column(connection, "feedback", "scan_id INTEGER REFERENCES scans(id) ON DELETE CASCADE")
        connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_scans_user_timestamp ON scans(user_id, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_feedback_user_timestamp ON feedback(user_id, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_sessions_user_expires ON auth_sessions(user_id, expires_at);
            INSERT OR IGNORE INTO schema_migrations(version) VALUES (1);
            """
        )
        connection.commit()
