from __future__ import annotations

import copy
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from backend.api.model_loader import (
    ModelService,
    ModelUnavailableError,
    _validate_onnx_contract,
    load_model_metadata,
)
from backend.api.routes import predict as predict_route
from backend.api.routes.predict import _validate_upload
from src.inference.preprocess_input import preprocess_for_onnx


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = PROJECT_ROOT / "models" / "releases" / "efficientnetv2_s_v1"
MODEL_PATH = RELEASE_DIR / "model.onnx"
METADATA_PATH = RELEASE_DIR / "model.json"
CANONICAL_MAPPING_PATH = PROJECT_ROOT / "data" / "class_mapping.json"


@pytest.fixture(scope="module")
def real_model_service() -> ModelService:
    service = ModelService(MODEL_PATH, METADATA_PATH)
    assert service.load(), service.load_error
    return service


def _release_payload() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _write_metadata(tmp_path: Path, payload: dict, name: str = "model.json") -> Path:
    output = tmp_path / name
    output.write_text(json.dumps(payload), encoding="utf-8")
    return output


def test_efficientnetv2_s_metadata_loading():
    metadata = load_model_metadata(METADATA_PATH)
    canonical = json.loads(CANONICAL_MAPPING_PATH.read_text(encoding="utf-8"))
    expected_order = [canonical["idx_to_class"][str(index)] for index in range(15)]

    assert metadata.architecture == "efficientnetv2_s"
    assert metadata.model_name == "EfficientNetV2-S"
    assert metadata.model_version == "v1"
    assert metadata.input_size == {"width": 300, "height": 300, "channels": 3}
    assert metadata.num_classes == 15
    assert [metadata.idx_to_class[index] for index in range(15)] == expected_order
    assert metadata.preprocessing["resize_mode"] == "shortest_center_crop"
    assert metadata.preprocessing["resize_strategy"] == "shortest_side"
    assert metadata.preprocessing["crop_strategy"] == "center_crop"
    assert metadata.preprocessing["interpolation"] == "bicubic"
    assert metadata.preprocessing["mean"] == [0.5, 0.5, 0.5]
    assert metadata.preprocessing["std"] == [0.5, 0.5, 0.5]
    assert metadata.temperature == pytest.approx(0.05)
    assert metadata.onnx_input_name == "images"
    assert metadata.onnx_output_name == "logits"


def test_efficientnetv2_s_preprocessing_output_shape_and_dtype():
    metadata = load_model_metadata(METADATA_PATH)
    image = Image.new("RGB", (640, 360), color=(255, 255, 255))
    output = preprocess_for_onnx(image, metadata.input_height, metadata.preprocessing)

    assert output.shape == (1, 3, 300, 300)
    assert output.dtype == np.float32
    assert np.allclose(output, 1.0)


def test_exif_orientation_is_corrected_before_inference():
    image = Image.new("RGB", (100, 200), color="green")
    exif = Image.Exif()
    exif[274] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    decoded = _validate_upload(buffer.getvalue(), "image/jpeg")
    assert decoded.mode == "RGB"
    assert decoded.size == (200, 100)


def test_class_count_and_class_order_validation(tmp_path):
    payload = _release_payload()

    wrong_count = copy.deepcopy(payload)
    wrong_count["num_classes"] = 14
    with pytest.raises(ValueError, match="idx_to_class"):
        load_model_metadata(_write_metadata(tmp_path, wrong_count, "wrong-count.json"))

    wrong_order = copy.deepcopy(payload)
    wrong_order["idx_to_class"]["0"], wrong_order["idx_to_class"]["1"] = (
        wrong_order["idx_to_class"]["1"],
        wrong_order["idx_to_class"]["0"],
    )
    with pytest.raises(ValueError, match="different class orders"):
        load_model_metadata(_write_metadata(tmp_path, wrong_order, "wrong-order.json"))


def test_onnx_output_class_dimension_must_match_metadata():
    metadata = load_model_metadata(METADATA_PATH)

    class ValueInfo:
        def __init__(self, name, shape, value_type="tensor(float)"):
            self.name = name
            self.shape = shape
            self.type = value_type

    class Session:
        def get_inputs(self):
            return [ValueInfo("images", ["batch", 3, 300, 300])]

        def get_outputs(self):
            return [ValueInfo("logits", ["batch", 14])]

    with pytest.raises(ValueError, match="output class dimension"):
        _validate_onnx_contract(Session(), metadata)


def test_invalid_or_missing_metadata_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_metadata(tmp_path / "missing.json")

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_model_metadata(invalid_path)

    service = ModelService(MODEL_PATH, tmp_path / "missing-serving-metadata.json")
    assert service.load() is False
    assert service.mode == "unavailable"
    with pytest.raises(ModelUnavailableError):
        service.predict(Image.new("RGB", (300, 300), color="green"))


def test_invalid_temperature_is_rejected(tmp_path):
    payload = _release_payload()
    payload["calibration"]["temperature"] = 0
    metadata_path = _write_metadata(tmp_path, payload)

    with pytest.raises(ValueError, match="finite and positive"):
        load_model_metadata(metadata_path)


def test_single_real_prediction_response_contract(real_model_service, client, monkeypatch):
    monkeypatch.setattr(predict_route, "model_service", real_model_service)
    image_path = next((PROJECT_ROOT / "data" / "processed" / "test").rglob("*.JPG"), None)
    if image_path is None:
        image_path = next((PROJECT_ROOT / "data" / "processed" / "test").rglob("*.jpg"))
    content = image_path.read_bytes()

    response = client.post(
        "/predict",
        files={"file": (image_path.name, content, "image/jpeg")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mock"] is False
    assert payload["mode"] == "onnx"
    assert payload["model_name"] == "EfficientNetV2-S"
    assert payload["model_version"] == "v1"
    assert payload["input_size"] == {"width": 300, "height": 300, "channels": 3}
    assert len(payload["top_3_predictions"]) == 3
    assert payload["scan_id"] > 0
    assert payload["detection_status"] in {"healthy", "disease_detected", "low_confidence", "review_recommended"}
    assert payload["class_name"] == payload["top_3_predictions"][0]["class_name"]
    assert 0.0 <= payload["confidence"] <= 1.0

    history_response = client.get("/history")
    assert history_response.status_code == 200
    assert history_response.json()[0]["predicted_class"] == payload["class_name"]
