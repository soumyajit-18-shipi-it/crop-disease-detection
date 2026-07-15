from __future__ import annotations

import json
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(__file__).resolve().parent / "disease_info.db"
MAPPING_PATH = PROJECT_ROOT / "data" / "class_mapping.json"

VERIFIED_ENTRIES = {
    "Tomato_Early_blight": (
        "Tomato",
        "Early blight",
        "Concentric brown lesions often begin on older leaves and may have yellow halos. Severe infections can cause lower leaf drop and expose fruit to sunscald.",
        "Remove infected leaves, improve airflow, avoid overhead watering, and follow local extension guidance for approved fungicides. Needs expert review for product choice and rates.",
        "moderate",
    ),
    "Tomato_Late_blight": (
        "Tomato",
        "Late blight",
        "Water-soaked lesions can expand quickly in cool humid weather, with dark stem lesions and possible white sporulation on leaf undersides. This disease can progress rapidly.",
        "Isolate or destroy infected plants and contact local extension services for region-specific late blight guidance. Needs expert review before chemical treatment decisions.",
        "severe",
    ),
    "Tomato_Leaf_Mold": (
        "Tomato",
        "Leaf mold",
        "Yellow patches appear on upper leaf surfaces with olive-gray mold on undersides in humid conditions. Dense canopies and poor ventilation increase risk.",
        "Increase ventilation, reduce humidity, prune dense foliage, and use resistant varieties where possible. Fungicide recommendations need expert review.",
        "moderate",
    ),
    "Potato_Early_blight": (
        "Potato",
        "Early blight",
        "Dark target-like leaf spots usually begin on older leaves. Defoliation can reduce tuber yield if disease pressure is high.",
        "Rotate crops, manage volunteer plants, maintain plant vigor, and seek local guidance for fungicide timing. Needs expert review for chemical recommendations.",
        "moderate",
    ),
    "Potato_Late_blight": (
        "Potato",
        "Late blight",
        "Irregular water-soaked lesions and white sporulation may appear on leaf undersides during humid weather. Tubers can develop brown rot after infection.",
        "Remove infected material, manage cull piles, avoid overhead irrigation, and consult local late-blight alerts. Needs expert review before pesticide use.",
        "severe",
    ),
    "Tomato_Healthy": (
        "Tomato",
        "Healthy",
        "Leaves appear green and uniform without obvious lesions or discoloration. Continue monitoring because early symptoms can be subtle.",
        "Maintain balanced watering, spacing, sanitation, and routine scouting.",
        "mild",
    ),
}


def _load_classes() -> list[str]:
    if not MAPPING_PATH.exists():
        return sorted(VERIFIED_ENTRIES)
    with MAPPING_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if "idx_to_class" in payload:
        return [payload["idx_to_class"][key] for key in sorted(payload["idx_to_class"], key=lambda x: int(x))]
    return [payload[key] for key in sorted(payload, key=lambda x: int(x))]


def _infer_crop_and_disease(class_name: str) -> tuple[str, str]:
    parts = class_name.replace("___", "_").split("_")
    crop = parts[0].replace("-", " ").title() if parts else "Unknown"
    disease = " ".join(parts[1:]).replace("-", " ").title() if len(parts) > 1 else class_name
    return crop, disease


def seed_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    classes = _load_classes()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DROP TABLE IF EXISTS diseases")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diseases (
                class_name TEXT PRIMARY KEY,
                crop TEXT,
                disease_name TEXT,
                symptoms TEXT NOT NULL,
                recommended_treatment TEXT NOT NULL,
                severity_level TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_hash TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                predicted_class TEXT NOT NULL,
                confidence REAL,
                message TEXT
            )
            """
        )
        for class_name in classes:
            crop, disease_name, symptoms, treatment, severity = VERIFIED_ENTRIES.get(
                class_name,
                (
                    *_infer_crop_and_disease(class_name),
                    (
                        "Placeholder disease description needs expert review. Visual symptoms should be validated "
                        "against local crop pathology guidance before use in field decisions."
                    ),
                    (
                        "Needs expert review. Do not make pesticide, disposal, or harvest decisions from this placeholder "
                        "entry alone; consult local agricultural extension or a qualified agronomist."
                    ),
                    "needs expert review",
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO diseases
                    (class_name, crop, disease_name, symptoms, recommended_treatment, severity_level)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (class_name, crop, disease_name, symptoms, treatment, severity),
            )
        conn.commit()


if __name__ == "__main__":
    seed_database()
    print(f"Seeded disease info database at {DB_PATH}")
