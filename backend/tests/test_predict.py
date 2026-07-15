from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app


client = TestClient(app)


def dummy_image_bytes(size=(32, 32)) -> bytes:
    image = Image.new("RGB", size, color=(80, 140, 80))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert "model_loaded" in payload
    assert "db_connected" in payload


def test_predict_accepts_valid_image_upload():
    files = {"file": ("leaf.jpg", dummy_image_bytes(), "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    payload = response.json()
    assert {
        "class_name",
        "confidence",
        "top_3_predictions",
        "symptoms",
        "recommended_treatment",
        "mode",
    } <= payload.keys()
    assert 0 <= payload["confidence"] <= 1
    assert len(payload["top_3_predictions"]) >= 1


def test_predict_rejects_invalid_file_type():
    files = {"file": ("notes.txt", b"not-an-image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400


def test_predict_rejects_unreadable_image():
    files = {"file": ("leaf.jpg", b"not-an-image", "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400


def test_predict_rejects_oversized_image(monkeypatch):
    from backend.api.routes import predict as predict_route

    monkeypatch.setattr(predict_route.settings, "max_upload_size_mb", 0)
    files = {"file": ("leaf.jpg", dummy_image_bytes(), "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 413
