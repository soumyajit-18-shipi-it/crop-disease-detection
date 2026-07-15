from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

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


def test_predict_accepts_valid_image_upload(monkeypatch):
    from backend.api.routes import predict as predict_route

    monkeypatch.setattr(predict_route.model_service, "predict", lambda image: {
        "class_name": "Tomato___Early_blight", "confidence": 0.9,
        "top_3_predictions": [{"class_name": "Tomato___Early_blight", "confidence": 0.9}],
        "mode": "onnx", "mock": False,
    })
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


def test_predict_returns_503_without_model(monkeypatch):
    from backend.api.model_loader import ModelUnavailableError
    from backend.api.routes import predict as predict_route

    monkeypatch.setattr(predict_route.model_service, "predict", lambda image: (_ for _ in ()).throw(ModelUnavailableError()))
    response = client.post("/predict", files={"file": ("leaf.jpg", dummy_image_bytes(), "image/jpeg")})
    assert response.status_code == 503


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


def test_model_service_applies_saved_temperature_to_confidence():
    from backend.api.model_loader import ModelService

    class Session:
        def run(self, _outputs, _inputs):
            return [np.array([[2.0, 0.0]], dtype=np.float32)]

    service = ModelService()
    service.session = Session()
    service.idx_to_class = {0: "healthy", 1: "diseased"}
    service.temperature = 2.0
    service.preprocessing = {
        "image_size": 16,
        "resize_mode": "stretch",
        "interpolation": "linear",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    }
    prediction = service.predict(Image.new("RGB", (20, 20), color="green"))
    assert prediction["class_name"] == "healthy"
    assert prediction["confidence"] == pytest.approx(0.7310586, rel=1e-5)
