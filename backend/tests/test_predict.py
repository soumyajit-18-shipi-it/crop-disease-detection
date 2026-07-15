from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

def dummy_image_bytes(size=(32, 32)) -> bytes:
    image = Image.new("RGB", size, color=(80, 140, 80))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_check_with_model_and_database_available(client, monkeypatch):
    from backend.api.routes import system as system_route

    available_model = SimpleNamespace(
        loaded=True,
        mode="onnx",
        model_name="EfficientNetV2-S",
        model_version="v1",
        input_size={"width": 300, "height": 300, "channels": 3},
    )
    monkeypatch.setattr(system_route, "model_service", available_model)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "model_loaded": True,
        "model_mode": "onnx",
        "db_connected": True,
        "model_name": "EfficientNetV2-S",
        "model_version": "v1",
        "input_size": {"width": 300, "height": 300, "channels": 3},
    }


def test_health_check_with_model_unavailable(client, monkeypatch):
    from backend.api.routes import system as system_route

    unavailable_model = SimpleNamespace(
        loaded=False,
        mode="unavailable",
        model_name=None,
        model_version=None,
        input_size=None,
    )
    monkeypatch.setattr(system_route, "model_service", unavailable_model)
    response = client.get("/health")
    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert payload["model_loaded"] is False
    assert payload["model_mode"] == "unavailable"
    assert payload["db_connected"] is True


def test_predict_accepts_valid_image_upload(client, monkeypatch):
    from backend.api.routes import predict as predict_route

    monkeypatch.setattr(predict_route.model_service, "predict", lambda image: {
        "class_name": "Tomato___Early_blight", "confidence": 0.9,
        "top_3_predictions": [{"class_name": "Tomato___Early_blight", "confidence": 0.9}],
        "mode": "onnx", "mock": False,
        "model_name": "EfficientNetV2-S", "model_version": "v1",
        "input_size": {"width": 300, "height": 300, "channels": 3},
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
    assert payload["mock"] is False
    assert payload["model_name"] == "EfficientNetV2-S"
    assert payload["input_size"] == {"width": 300, "height": 300, "channels": 3}


def test_predict_returns_503_without_model(client, monkeypatch):
    from backend.api.model_loader import ModelUnavailableError
    from backend.api.routes import predict as predict_route

    monkeypatch.setattr(predict_route.model_service, "predict", lambda image: (_ for _ in ()).throw(ModelUnavailableError()))
    response = client.post("/predict", files={"file": ("leaf.jpg", dummy_image_bytes(), "image/jpeg")})
    assert response.status_code == 503


def test_predict_rejects_invalid_file_type(client):
    files = {"file": ("notes.txt", b"not-an-image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400


def test_predict_rejects_unreadable_image(client):
    files = {"file": ("leaf.jpg", b"not-an-image", "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400


def test_predict_rejects_oversized_image(client, monkeypatch):
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
    service.metadata = SimpleNamespace(
        architecture="test",
        model_name="Test model",
        model_version="test-v1",
        input_size={"width": 16, "height": 16, "channels": 3},
    )
    service.input_name = "images"
    service.output_name = "logits"
    service.image_size = 16
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
