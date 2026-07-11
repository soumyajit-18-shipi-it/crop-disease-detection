import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.api.schemas import DiseaseInfo


router = APIRouter(tags=["disease-info"])
DB_PATH = Path(__file__).resolve().parents[2] / "db" / "disease_info.db"


def get_disease_info_by_class(class_name: str) -> dict[str, str]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT class_name, symptoms, recommended_treatment FROM diseases WHERE class_name = ?",
            (class_name,),
        ).fetchone()

    if row:
        return dict(row)

    return {
        "class_name": class_name,
        "symptoms": "No placeholder symptoms are available for this class yet.",
        "recommended_treatment": "Review the plant manually and add disease guidance to the SQLite database.",
    }


@router.get("/disease/{class_name}", response_model=DiseaseInfo)
def disease_info(class_name: str) -> DiseaseInfo:
    disease = get_disease_info_by_class(class_name)
    if disease["symptoms"].startswith("No placeholder"):
        raise HTTPException(status_code=404, detail=f"No disease info found for {class_name}.")
    return DiseaseInfo(**disease)
