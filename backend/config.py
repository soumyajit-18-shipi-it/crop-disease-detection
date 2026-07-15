from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_DIR = PROJECT_ROOT / "models" / "releases" / "efficientnetv2_s_v1"


def resolve_project_path(value: str | Path) -> Path:
    """Resolve relative configuration paths from the repository root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _split_origins(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
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

    def __post_init__(self) -> None:
        if self.cors_origins is None:
            self.cors_origins = _split_origins(
                os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
            )

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


settings = Settings()
