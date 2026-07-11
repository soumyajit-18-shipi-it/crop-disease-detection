import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "disease_info.db"

DISEASES = [
    (
        "Tomato_Early_blight",
        "Concentric brown spots on older leaves, yellowing around lesions, and gradual leaf drop.",
        "Remove infected leaves, improve airflow, avoid overhead watering, and apply a labeled copper or chlorothalonil fungicide when pressure is high.",
    ),
    (
        "Tomato_Late_blight",
        "Water-soaked lesions, pale green leaf patches, dark stem spots, and rapid collapse in cool humid weather.",
        "Remove infected plants, avoid working wet foliage, improve spacing, and use a late-blight labeled fungicide program.",
    ),
    (
        "Tomato_Leaf_Mold",
        "Yellow patches on upper leaf surfaces with olive-gray mold growth on undersides.",
        "Increase ventilation, reduce humidity, prune dense foliage, and use resistant varieties or labeled fungicides.",
    ),
    (
        "Potato_Early_blight",
        "Dark target-like leaf spots, chlorosis, and premature defoliation beginning on older leaves.",
        "Rotate crops, remove volunteer potatoes, maintain plant vigor, and apply protectant fungicides if disease develops.",
    ),
    (
        "Potato_Late_blight",
        "Irregular water-soaked leaf lesions, white sporulation on leaf undersides, and brown tuber rot risk.",
        "Destroy infected plants, manage cull piles, avoid overhead irrigation, and follow local late-blight fungicide guidance.",
    ),
    (
        "Tomato_Healthy",
        "Leaves appear green and uniform with no obvious disease lesions.",
        "Continue routine monitoring, balanced watering, proper spacing, and sanitation to reduce future disease pressure.",
    ),
]


def seed_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diseases (
                class_name TEXT PRIMARY KEY,
                symptoms TEXT NOT NULL,
                recommended_treatment TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO diseases (class_name, symptoms, recommended_treatment)
            VALUES (?, ?, ?)
            """,
            DISEASES,
        )
        conn.commit()


if __name__ == "__main__":
    seed_database()
    print(f"Seeded disease info database at {DB_PATH}")
