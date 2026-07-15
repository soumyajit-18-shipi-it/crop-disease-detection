from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from src.inference.preprocess_input import preprocess_for_onnx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX_PATH = PROJECT_ROOT / "models" / "onnx" / "model.onnx"
logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


class ModelService:
    def __init__(self) -> None:
        self.session = None
        self.input_name = "images"
        self.output_name = "logits"
        self.image_size = 224
        self.preprocessing: dict = {}
        self.temperature = 1.0
        self.idx_to_class: dict[int, str] = {}
        self.model_path: Path | None = None
        self.load_error: str | None = None

    def load(self) -> None:
        model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_ONNX_PATH))).resolve()
        metadata_override = os.getenv("MODEL_METADATA_PATH")
        if metadata_override:
            metadata_path = Path(metadata_override).resolve()
        else:
            candidates = [model_path.with_suffix(".json"), model_path.parent / "metadata.json"]
            metadata_path = next((path for path in candidates if path.is_file()), candidates[0]).resolve()
        self.session = None
        self.load_error = None
        self.preprocessing = {}
        self.temperature = 1.0
        self.idx_to_class = {}
        self.model_path = None
        if not model_path.is_file() or not metadata_path.is_file():
            self.load_error = f"ONNX model bundle is incomplete: {model_path} and {metadata_path} are required"
            logger.error(self.load_error)
            return
        try:
            import onnxruntime as ort
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            mapping = metadata.get("idx_to_class", {})
            if not mapping:
                raise ValueError("Model metadata has no idx_to_class mapping")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            available = set(ort.get_available_providers())
            selected = [provider for provider in providers if provider in available]
            self.session = ort.InferenceSession(str(model_path), providers=selected)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.idx_to_class = {int(index): str(label) for index, label in mapping.items()}
            self.image_size = int(metadata["image_size"])
            self.preprocessing = metadata.get("preprocessing", {})
            self.temperature = float(metadata.get("calibration", {}).get("temperature", 1.0))
            if not np.isfinite(self.temperature) or self.temperature <= 0:
                raise ValueError("Model metadata has an invalid calibration temperature")
            self.model_path = model_path
            logger.info("Loaded ONNX model %s with providers %s", model_path, self.session.get_providers())
        except Exception as exc:
            self.session = None
            self.load_error = f"Failed to load ONNX model: {exc}"
            logger.exception(self.load_error)

    @property
    def loaded(self) -> bool:
        return self.session is not None

    @property
    def mode(self) -> str:
        return "onnx" if self.loaded else "unavailable"

    def predict(self, image: Image.Image) -> dict:
        if not self.loaded:
            raise ModelUnavailableError(self.load_error or "ONNX model is not loaded")
        model_input = preprocess_for_onnx(image, self.image_size, self.preprocessing)
        logits = self.session.run([self.output_name], {self.input_name: model_input})[0]
        probabilities = self._softmax(logits / self.temperature)[0]
        top_indices = probabilities.argsort()[-min(3, len(probabilities)):][::-1]
        top = [
            {"class_name": self.idx_to_class[int(index)], "confidence": float(probabilities[index])}
            for index in top_indices
        ]
        return {
            "class_name": top[0]["class_name"], "confidence": top[0]["confidence"],
            "top_3_predictions": top, "mode": "onnx", "mock": False,
        }

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)


model_service = ModelService()


def load_model() -> None:
    model_service.load()


def get_supported_classes() -> list[str]:
    return [model_service.idx_to_class[index] for index in sorted(model_service.idx_to_class)]
