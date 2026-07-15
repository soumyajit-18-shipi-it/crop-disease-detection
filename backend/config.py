from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / "backend" / ".env", override=False)
DEFAULT_RELEASE_DIR = PROJECT_ROOT / "models" / "releases" / "efficientnetv2_s_v1"


def resolve_project_path(value: str | Path) -> Path:
    """Resolve relative configuration paths from the repository root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _split_origins(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development").strip().lower()
    )
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "").strip(),
        repr=False,
    )
    database_pool_min_size: int = field(
        default_factory=lambda: int(os.getenv("DATABASE_POOL_MIN_SIZE", "1"))
    )
    database_pool_max_size: int = field(
        default_factory=lambda: int(os.getenv("DATABASE_POOL_MAX_SIZE", "5"))
    )
    database_connect_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("DATABASE_CONNECT_TIMEOUT_SECONDS", "15"))
    )
    model_path: Path = field(
        default_factory=lambda: resolve_project_path(
            os.getenv("MODEL_PATH", str(DEFAULT_RELEASE_DIR / "model.onnx"))
        )
    )
    model_metadata_path: Path = field(
        default_factory=lambda: resolve_project_path(
            os.getenv("MODEL_METADATA_PATH", str(DEFAULT_RELEASE_DIR / "model.json"))
        )
    )
    model_release_manifest_path: Path = field(
        default_factory=lambda: resolve_project_path(
            os.getenv("MODEL_RELEASE_MANIFEST", str(DEFAULT_RELEASE_DIR / "release.json"))
        )
    )
    db_path: Path = field(
        default_factory=lambda: resolve_project_path(
            os.getenv("DB_PATH", "backend/db/disease_info.db")
        )
    )
    cors_origins: list[str] | None = None
    max_upload_size_mb: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    )
    log_dir: Path = field(
        default_factory=lambda: resolve_project_path(os.getenv("LOG_DIR", "backend/logs"))
    )
    log_to_file: bool = field(default_factory=lambda: _as_bool(os.getenv("LOG_TO_FILE", "true")))
    app_url: str = field(default_factory=lambda: os.getenv("APP_URL", "").rstrip("/"))
    oauth_callback_url: str = field(default_factory=lambda: os.getenv("OAUTH_CALLBACK_URL", "").strip())
    google_client_id: str = field(default_factory=lambda: os.getenv("GOOGLE_CLIENT_ID", "").strip())
    google_client_secret: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CLIENT_SECRET", "").strip(),
        repr=False,
    )
    auth_secret: str = field(
        default_factory=lambda: os.getenv("AUTH_SECRET", "").strip(),
        repr=False,
    )
    session_ttl_hours: int = field(default_factory=lambda: int(os.getenv("SESSION_TTL_HOURS", "168")))
    cookie_secure: bool = field(default_factory=lambda: _as_bool(os.getenv("COOKIE_SECURE", "true")))
    cookie_samesite: str = field(default_factory=lambda: os.getenv("COOKIE_SAMESITE", "lax").strip().lower())
    cookie_domain: str | None = field(
        default_factory=lambda: os.getenv("COOKIE_DOMAIN", "").strip() or None
    )
    low_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.60"))
    )

    def __post_init__(self) -> None:
        if self.environment not in {"development", "test", "production"}:
            raise ValueError("ENVIRONMENT must be development, test, or production.")
        if self.database_pool_min_size < 1:
            raise ValueError("DATABASE_POOL_MIN_SIZE must be at least 1.")
        if self.database_pool_max_size < self.database_pool_min_size:
            raise ValueError("DATABASE_POOL_MAX_SIZE must be at least DATABASE_POOL_MIN_SIZE.")
        if self.database_connect_timeout_seconds < 1:
            raise ValueError("DATABASE_CONNECT_TIMEOUT_SECONDS must be at least 1.")
        if self.cors_origins is None:
            self.cors_origins = _split_origins(
                os.getenv("CORS_ORIGINS", "http://127.0.0.1:5173")
            )
        if "*" in self.cors_origins:
            raise ValueError("CORS_ORIGINS cannot contain '*' when cookie authentication is enabled.")
        if self.cookie_samesite not in {"lax", "strict", "none"}:
            raise ValueError("COOKIE_SAMESITE must be lax, strict, or none.")
        if self.cookie_samesite == "none" and not self.cookie_secure:
            raise ValueError("COOKIE_SAMESITE=none requires COOKIE_SECURE=true.")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def google_auth_configured(self) -> bool:
        return bool(
            self.google_client_id
            and self.google_client_secret
            and self.auth_secret
            and self.app_url
            and self.oauth_callback_url
        )


settings = Settings()
