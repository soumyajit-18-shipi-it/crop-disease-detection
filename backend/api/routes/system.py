from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from backend.api.auth import AuthContext, require_csrf, require_user
from backend.api.model_loader import ModelUnavailableError, get_supported_classes, model_service
from backend.api.routes.disease_info import db_connect
from backend.api.schemas import DashboardSummary, FeedbackRequest, HealthResponse, ScanHistoryItem
from backend.db.database import timestamp_string


router = APIRouter(tags=["system"])


def _scan_from_row(row) -> ScanHistoryItem:
    payload = dict(row)
    raw_warnings = payload.get("quality_warnings")
    if isinstance(raw_warnings, list):
        payload["quality_warnings"] = raw_warnings
    else:
        try:
            payload["quality_warnings"] = json.loads(raw_warnings) if raw_warnings else []
        except (TypeError, json.JSONDecodeError):
            payload["quality_warnings"] = []
    payload["timestamp"] = timestamp_string(payload.get("timestamp"))
    return ScanHistoryItem(**payload)


@router.get("/health", response_model=HealthResponse)
def health_check(response: Response) -> HealthResponse:
    db_connected = True
    try:
        with db_connect() as connection:
            connection.execute("SELECT 1").fetchone()
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
def history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    search: str | None = Query(default=None, max_length=100),
    auth: AuthContext = Depends(require_user),
) -> list[ScanHistoryItem]:
    parameters: list[object] = [auth.user_id]
    search_clause = ""
    if search and search.strip():
        search_clause = "AND predicted_class LIKE ? ESCAPE '\\'"
        escaped = search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        parameters.append(f"%{escaped}%")
    parameters.extend([limit, offset])
    with db_connect() as connection:
        rows = connection.execute(
            f"""
            SELECT id, timestamp, predicted_class, confidence, image_hash,
                   original_filename, content_type, file_size, model_name,
                   model_version, detection_status, quality_status, quality_warnings
            FROM scans
            WHERE user_id = ? {search_clause}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            tuple(parameters),
        ).fetchall()
    return [_scan_from_row(row) for row in rows]


@router.get("/dashboard", response_model=DashboardSummary)
def dashboard(auth: AuthContext = Depends(require_user)) -> DashboardSummary:
    with db_connect() as connection:
        totals = connection.execute(
            """
            SELECT COUNT(*) AS total_scans,
                   SUM(CASE WHEN LOWER(predicted_class) LIKE '%healthy%' THEN 1 ELSE 0 END) AS healthy_scans,
                   SUM(CASE WHEN LOWER(predicted_class) NOT LIKE '%healthy%' THEN 1 ELSE 0 END) AS diseased_scans,
                   SUM(CASE WHEN detection_status = 'low_confidence' THEN 1 ELSE 0 END) AS low_confidence_scans,
                   AVG(confidence) AS average_confidence,
                   MAX(timestamp) AS latest_scan_at,
                   COUNT(DISTINCT CASE WHEN LOWER(predicted_class) NOT LIKE '%healthy%' THEN predicted_class END)
                       AS active_disease_classes
            FROM scans WHERE user_id = ?
            """,
            (auth.user_id,),
        ).fetchone()
        distribution_rows = connection.execute(
            """
            SELECT predicted_class AS class_name, COUNT(*) AS count
            FROM scans
            WHERE user_id = ? AND LOWER(predicted_class) NOT LIKE '%healthy%'
            GROUP BY predicted_class
            ORDER BY count DESC, predicted_class ASC
            """,
            (auth.user_id,),
        ).fetchall()
        recent_rows = connection.execute(
            """
            SELECT id, timestamp, predicted_class, confidence, image_hash,
                   original_filename, content_type, file_size, model_name,
                   model_version, detection_status, quality_status, quality_warnings
            FROM scans WHERE user_id = ? ORDER BY id DESC LIMIT 5
            """,
            (auth.user_id,),
        ).fetchall()
    total = int(totals["total_scans"] or 0)
    healthy = int(totals["healthy_scans"] or 0)
    diseased = int(totals["diseased_scans"] or 0)
    distribution_total = sum(int(row["count"]) for row in distribution_rows)
    return DashboardSummary(
        total_scans=total,
        healthy_scans=healthy,
        diseased_scans=diseased,
        low_confidence_scans=int(totals["low_confidence_scans"] or 0),
        average_confidence=float(totals["average_confidence"]) if totals["average_confidence"] is not None else None,
        healthy_percentage=(healthy / total * 100.0) if total else None,
        active_disease_classes=int(totals["active_disease_classes"] or 0),
        latest_scan_at=timestamp_string(totals["latest_scan_at"]),
        disease_distribution=[
            {
                "class_name": str(row["class_name"]),
                "count": int(row["count"]),
                "percentage": (int(row["count"]) / distribution_total * 100.0) if distribution_total else 0.0,
            }
            for row in distribution_rows
        ],
        recent_scans=[_scan_from_row(row) for row in recent_rows],
    )


@router.post("/feedback")
def feedback(
    payload: FeedbackRequest,
    auth: AuthContext = Depends(require_csrf),
) -> dict[str, str]:
    with db_connect() as connection:
        predicted_class = payload.predicted_class
        confidence = payload.confidence
        if payload.scan_id is not None:
            scan = connection.execute(
                "SELECT predicted_class, confidence FROM scans WHERE id = ? AND user_id = ?",
                (payload.scan_id, auth.user_id),
            ).fetchone()
            if not scan:
                raise HTTPException(status_code=404, detail="Scan not found.")
            predicted_class = str(scan["predicted_class"])
            confidence = float(scan["confidence"])
        if not predicted_class:
            raise HTTPException(status_code=422, detail="scan_id or predicted_class is required.")
        connection.execute(
            """
            INSERT INTO feedback(user_id, scan_id, predicted_class, confidence, message)
            VALUES (?, ?, ?, ?, ?)
            """,
            (auth.user_id, payload.scan_id, predicted_class, confidence, payload.message),
        )
        connection.commit()
    return {"status": "received"}
