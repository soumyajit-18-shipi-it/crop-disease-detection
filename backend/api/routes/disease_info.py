from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.api.schemas import DiseaseInfo


router = APIRouter(tags=["disease-info"])
DB_PATH = Path(settings.db_path)


def db_connect(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


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
        return dict(row)

    crop = class_name.split("_")[0].title() if class_name else None
    return {
        "class_name": class_name,
        "crop": crop,
        "disease_name": class_name.replace("_", " "),
        "symptoms": "No reviewed symptoms are available for this class yet.",
        "recommended_treatment": "Needs expert review before field use.",
        "severity_level": "needs expert review",
    }


@router.get("/disease/{class_name}", response_model=DiseaseInfo)
def disease_info(class_name: str) -> DiseaseInfo:
    disease = get_disease_info_by_class(class_name)
    if disease["symptoms"].startswith("No reviewed"):
        raise HTTPException(status_code=404, detail=f"No disease info found for {class_name}.")
    return DiseaseInfo(**disease)
