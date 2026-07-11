from __future__ import annotations

from fastapi import APIRouter

from backend.api.model_loader import get_supported_classes, model_service
from backend.api.routes.disease_info import db_connect
from backend.api.schemas import FeedbackRequest, HealthResponse, ScanHistoryItem


router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    db_connected = True
    try:
        with db_connect() as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception:
        db_connected = False
    return HealthResponse(
        status="ok" if db_connected else "degraded",
        model_loaded=model_service.loaded,
        model_mode=model_service.mode,
        db_connected=db_connected,
    )


@router.get("/classes", response_model=list[str])
def classes() -> list[str]:
    return get_supported_classes()


@router.get("/history", response_model=list[ScanHistoryItem])
def history(limit: int = 50) -> list[ScanHistoryItem]:
    limit = max(1, min(limit, 200))
    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, predicted_class, confidence, image_hash
            FROM scans
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [ScanHistoryItem(**dict(row)) for row in rows]


@router.post("/feedback")
def feedback(payload: FeedbackRequest) -> dict[str, str]:
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO feedback (predicted_class, confidence, message) VALUES (?, ?, ?)",
            (payload.predicted_class, payload.confidence, payload.message),
        )
        conn.commit()
    return {"status": "received"}
