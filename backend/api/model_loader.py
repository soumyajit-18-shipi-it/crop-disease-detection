from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_MAPPING_PATH = PROJECT_ROOT / "data" / "class_mapping.json"
DEFAULT_ONNX_PATH = PROJECT_ROOT / "models" / "onnx" / "model.onnx"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_config.json"

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self) -> None:
        self.mode = "mock"
        self.session = None
        self.input_name = "input"
        self.image_size = 224
        self.idx_to_class = self._load_idx_to_class()

    def _load_idx_to_class(self) -> dict[int, str]:
        if not CLASS_MAPPING_PATH.exists():
            return {0: "Unknown"}
        with CLASS_MAPPING_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if "idx_to_class" in payload:
            return {int(k): str(v) for k, v in payload["idx_to_class"].items()}
        return {int(k): str(v) for k, v in payload.items()}

    def load(self) -> None:
        model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_ONNX_PATH)))
        if not model_path.exists():
            logger.warning("No ONNX model found at %s. API will run in mock prediction mode.", model_path)
            self.mode = "mock"
            return
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            metadata_path = model_path.with_suffix(".json")
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as file:
                    metadata = json.load(file)
                self.idx_to_class = {int(k): str(v) for k, v in metadata.get("idx_to_class", {}).items()} or self.idx_to_class
                self.image_size = int(metadata.get("image_size", self.image_size))
            self.mode = "onnx"
            logger.info("Loaded ONNX model from %s", model_path)
        except Exception as exc:
            logger.warning("Failed to load ONNX model; falling back to mock mode: %s", exc)
            self.mode = "mock"
            self.session = None

    @property
    def loaded(self) -> bool:
        return self.mode == "onnx" and self.session is not None

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((self.image_size, self.image_size))
        arr = np.asarray(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        return arr.astype(np.float32)

    def predict(self, image: Image.Image) -> dict:
        if not self.loaded:
            return self.mock_predict()
        logits = self.session.run(None, {self.input_name: self._preprocess(image)})[0]
        probs = self._softmax(logits)[0]
        top_indices = probs.argsort()[-3:][::-1]
        top_3 = [
            {"class_name": self.idx_to_class.get(int(idx), f"class_{idx}"), "confidence": float(probs[idx])}
            for idx in top_indices
        ]
        return {
            "class_name": top_3[0]["class_name"],
            "confidence": top_3[0]["confidence"],
            "top_3_predictions": top_3,
            "mode": self.mode,
        }

    def mock_predict(self) -> dict:
        class_name = random.choice(list(self.idx_to_class.values()))
        confidence = round(random.uniform(0.55, 0.82), 4)
        alternatives = random.sample(list(self.idx_to_class.values()), min(3, len(self.idx_to_class)))
        if class_name not in alternatives:
            alternatives[0] = class_name
        top_3 = [
            {"class_name": name, "confidence": max(0.01, round(confidence - (i * 0.12), 4))}
            for i, name in enumerate(alternatives)
        ]
        return {"class_name": class_name, "confidence": confidence, "top_3_predictions": top_3, "mode": "mock"}

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)


model_service = ModelService()


def load_model() -> None:
    model_service.load()


def get_supported_classes() -> list[str]:
    return [model_service.idx_to_class[idx] for idx in sorted(model_service.idx_to_class)]


def mock_predict(image_bytes: bytes | None = None, filename: str | None = None) -> dict:
    return model_service.mock_predict()
