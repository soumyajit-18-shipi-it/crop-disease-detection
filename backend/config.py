from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _split_origins(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    model_path: str = os.getenv("MODEL_PATH", "models/onnx/model.onnx")
    db_path: str = os.getenv("DB_PATH", "backend/db/disease_info.db")
    cors_origins: list[str] = None
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    log_dir: Path = Path("backend/logs")

    def __post_init__(self) -> None:
        if self.cors_origins is None:
            self.cors_origins = _split_origins(
                os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
            )

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


settings = Settings()
