import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_MAPPING_PATH = PROJECT_ROOT / "data" / "class_mapping.json"


def load_class_mapping(path: Path = CLASS_MAPPING_PATH) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def mock_predict(image_bytes: bytes, filename: str | None = None) -> dict[str, float | str]:
    """Return a mock disease prediction.

    Replace this function body with real PyTorch preprocessing/model inference later.
    Keep the returned keys stable so API and frontend code do not need to change.
    """
    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    class_mapping = load_class_mapping()
    class_name = random.choice(list(class_mapping.values()))
    confidence = round(random.uniform(0.72, 0.98), 4)
    return {"class_name": class_name, "confidence": confidence}
