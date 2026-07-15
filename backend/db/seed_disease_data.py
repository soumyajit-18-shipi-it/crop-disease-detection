from __future__ import annotations

from pathlib import Path

from backend.db.database import (
    connect_database,
    database_backend,
    migrate_database,
    validate_database_configuration,
)


DB_PATH = Path(__file__).resolve().parent / "disease_info.db"

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
    "Potato___Early_blight": (
        "Potato",
        "Early blight",
        "Dark target-like leaf spots usually begin on older leaves. Defoliation can reduce tuber yield if disease pressure is high.",
        "Rotate crops, manage volunteer plants, maintain plant vigor, and seek local guidance for fungicide timing. Needs expert review for chemical recommendations.",
        "moderate",
    ),
    "Potato___Late_blight": (
        "Potato",
        "Late blight",
        "Irregular water-soaked lesions and white sporulation may appear on leaf undersides during humid weather. Tubers can develop brown rot after infection.",
        "Remove infected material, manage cull piles, avoid overhead irrigation, and consult local late-blight alerts. Needs expert review before pesticide use.",
        "severe",
    ),
    "Tomato_healthy": (
        "Tomato",
        "Healthy",
        "Leaves appear green and uniform without obvious lesions or discoloration. Continue monitoring because early symptoms can be subtle.",
        "Maintain balanced watering, spacing, sanitation, and routine scouting.",
        "mild",
    ),
}


def seed_database(db_path: str | Path | None = None) -> None:
    if db_path is not None:
        target_path = Path(db_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        migrate_database(target_path)
        connection_context = connect_database(target_path)
    else:
        validate_database_configuration()
        connection_context = connect_database()
    with connection_context as conn:
        placeholders = ",".join("?" for _ in VERIFIED_ENTRIES)
        conn.execute(
            f"DELETE FROM diseases WHERE class_name NOT IN ({placeholders})",
            tuple(VERIFIED_ENTRIES),
        )
        for class_name, entry in VERIFIED_ENTRIES.items():
            crop, disease_name, symptoms, treatment, severity = entry
            conn.execute(
                """
                INSERT INTO diseases
                    (class_name, crop, disease_name, symptoms, recommended_treatment, severity_level)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(class_name) DO UPDATE SET
                    crop = excluded.crop,
                    disease_name = excluded.disease_name,
                    symptoms = excluded.symptoms,
                    recommended_treatment = excluded.recommended_treatment,
                    severity_level = excluded.severity_level
                """,
                (class_name, crop, disease_name, symptoms, treatment, severity),
            )
        conn.commit()


if __name__ == "__main__":
    validate_database_configuration()
    seed_database()
    print(f"Seeded verified disease metadata using {database_backend()}.")
