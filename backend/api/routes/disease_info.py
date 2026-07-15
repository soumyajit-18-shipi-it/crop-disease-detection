from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.api.schemas import DiseaseInfo
from backend.db.database import connect_database


router = APIRouter(tags=["disease-info"])


def db_connect(path: str | Path | None = None):
    """Backward-compatible database connection helper."""
    return connect_database(Path(path)) if path is not None else connect_database()


def get_disease_info_by_class(class_name: str) -> dict:
    with db_connect() as conn:
        row = conn.execute(
            """
            SELECT class_name, crop, disease_name, symptoms, recommended_treatment, severity_level
            FROM diseases WHERE class_name = ?
            """,
            (class_name,),
        ).fetchone()

    if row:
        return {**dict(row), "information_status": "reviewed"}

    normalized = class_name.replace("___", "_").replace("__", "_")
    parts = [part for part in normalized.split("_") if part]
    crop = parts[0].replace("-", " ").title() if parts else None
    disease_name = " ".join(parts[1:]).replace("-", " ").title() if len(parts) > 1 else None
    return {
        "class_name": class_name,
        "crop": crop,
        "disease_name": disease_name,
        "symptoms": None,
        "recommended_treatment": None,
        "severity_level": None,
        "information_status": "unavailable",
    }


@router.get("/disease/{class_name}", response_model=DiseaseInfo)
def disease_info(class_name: str) -> DiseaseInfo:
    disease = get_disease_info_by_class(class_name)
    if disease["symptoms"] is None:
        raise HTTPException(status_code=404, detail=f"No disease info found for {class_name}.")
    return DiseaseInfo(**disease, information_status="reviewed")
