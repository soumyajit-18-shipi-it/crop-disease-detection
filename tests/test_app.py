import io

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["service"] == "Crop Disease Detection API"


def test_predict_endpoint_success() -> None:
    file_content = b"fake-image-bytes"
    file = io.BytesIO(file_content)
    response = client.post("/predict", files={"file": ("tomato_leaf.jpg", file, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "tomato_leaf.jpg"
    assert data["prediction"] == "Tomato Early Blight"
    assert data["confidence"] == 0.94


def test_predict_endpoint_no_filename() -> None:
    file_content = b"fake-image-bytes"
    file = io.BytesIO(file_content)
    response = client.post("/predict", files={"file": ("", file, "image/jpeg")})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
