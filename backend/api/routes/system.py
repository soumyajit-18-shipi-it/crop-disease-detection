from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from backend.api.model_loader import ModelUnavailableError, get_supported_classes, model_service
from backend.api.routes.disease_info import db_connect
from backend.api.schemas import FeedbackRequest, HealthResponse, ScanHistoryItem


router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health_check(response: Response) -> HealthResponse:
    db_connected = True
    try:
        with db_connect() as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception:
        db_connected = False
    ready = db_connected and model_service.loaded
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if ready else ("unavailable" if not model_service.loaded else "degraded"),
        model_loaded=model_service.loaded,
        model_mode=model_service.mode,
        db_connected=db_connected,
        model_name=model_service.model_name,
        model_version=model_service.model_version,
        input_size=model_service.input_size,
    )


@router.get("/classes", response_model=list[str])
def classes() -> list[str]:
    try:
        return get_supported_classes()
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The inference model is not available.",
        ) from exc


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
