from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_accepts_image_upload():
    files = {"file": ("leaf.jpg", b"fake-image-bytes", "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    payload = response.json()
    assert {"class_name", "confidence", "symptoms", "recommended_treatment"} <= payload.keys()
    assert 0 <= payload["confidence"] <= 1
