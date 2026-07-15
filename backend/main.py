from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.api.model_loader import load_model
from backend.api.routes.auth import router as auth_router
from backend.api.routes.disease_info import router as disease_info_router
from backend.api.routes.predict import router as predict_router
from backend.api.routes.system import router as system_router
from backend.config import settings
from backend.db.database import (
    close_database_pool,
    database_backend,
    migrate_database,
    validate_database_configuration,
)
from backend.db.seed_disease_data import seed_database


def configure_logging() -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if settings.log_to_file:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(settings.log_dir / "requests.log", maxBytes=2_000_000, backupCount=5)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        handlers.insert(0, handler)
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    # The application middleware logs sanitized paths and statuses. Uvicorn's
    # default access line includes OAuth callback query strings (code/state).
    logging.getLogger("uvicorn.access").disabled = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    backend = validate_database_configuration()
    if backend == "sqlite":
        migrate_database()
    seed_database()
    load_model()
    try:
        yield
    finally:
        close_database_pool()


app = FastAPI(title="Crop Disease Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-CSRF-Token"],
)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(self)")
    duration_ms = (time.perf_counter() - start) * 1000
    logging.info(
        "method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


app.include_router(system_router)
app.include_router(auth_router)
app.include_router(predict_router)
app.include_router(disease_info_router)
