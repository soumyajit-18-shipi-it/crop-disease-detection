from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from backend.config import resolve_project_path, settings
from src.inference.model_release import (
    ModelRelease,
    load_release_manifest,
    verify_asset,
    verify_supporting_assets,
)
from src.inference.preprocess_input import preprocess_for_onnx


logger = logging.getLogger(__name__)
SUPPORTED_RESIZE_MODES = {"stretch", "shortest_center_crop"}
SUPPORTED_INTERPOLATIONS = {
    "nearest",
    "linear",
    "bilinear",
    "bicubic",
    "cubic",
    "area",
    "lanczos",
}


class ModelUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelMetadata:
    architecture: str
    model_name: str
    model_version: str
    input_width: int
    input_height: int
    channels: int
    layout: str
    dtype: str
    num_classes: int
    idx_to_class: dict[int, str]
    preprocessing: dict
    temperature: float
    onnx_input_name: str
    onnx_output_name: str

    @property
    def input_size(self) -> dict[str, int]:
        return {
            "width": self.input_width,
            "height": self.input_height,
            "channels": self.channels,
        }


def _required_mapping(payload: Mapping, key: str) -> Mapping:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Model metadata field '{key}' must be an object")
    return value


def _required_string(payload: Mapping, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Model metadata field '{key}' must be a non-empty string")
    return value.strip()


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Model metadata field '{field_name}' must be a positive integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Model metadata field '{field_name}' must be a positive integer"
        ) from exc
    if converted <= 0:
        raise ValueError(f"Model metadata field '{field_name}' must be a positive integer")
    return converted


def _three_channel_values(preprocessing: Mapping, key: str, *, positive: bool = False) -> list[float]:
    values = preprocessing.get(key)
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"Model preprocessing field '{key}' must contain three values")
    converted = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(converted)) or (positive and np.any(converted <= 0)):
        qualifier = "finite positive" if positive else "finite"
        raise ValueError(f"Model preprocessing field '{key}' must contain {qualifier} values")
    return [float(value) for value in converted]


def load_model_metadata(metadata_path: str | Path) -> ModelMetadata:
    """Load and strictly validate the serving contract for an ONNX release."""
    path = resolve_project_path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model metadata does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Model metadata is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Model metadata root must be an object")

    architecture = _required_string(payload, "architecture")
    model_name = _required_string(payload, "model_name")
    model_version = _required_string(payload, "model_version")
    num_classes = _positive_integer(payload.get("num_classes"), "num_classes")

    input_spec = _required_mapping(payload, "input")
    input_width = _positive_integer(input_spec.get("width"), "input.width")
    input_height = _positive_integer(input_spec.get("height"), "input.height")
    channels = _positive_integer(input_spec.get("channels"), "input.channels")
    layout = _required_string(input_spec, "layout").upper()
    dtype = _required_string(input_spec, "dtype").lower()
    if channels != 3:
        raise ValueError("The image serving contract requires exactly three RGB channels")
    if layout != "NCHW":
        raise ValueError("The ONNX image input layout must be NCHW")
    if dtype != "float32":
        raise ValueError("The ONNX image input dtype must be float32")
    if input_width != input_height:
        raise ValueError("The current image preprocessor requires a square model input")
    image_size = _positive_integer(payload.get("image_size"), "image_size")
    if image_size != input_width or image_size != input_height:
        raise ValueError("image_size does not match the explicit input dimensions")

    raw_mapping = _required_mapping(payload, "idx_to_class")
    try:
        idx_to_class = {int(index): str(label) for index, label in raw_mapping.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("idx_to_class keys must be integer indices") from exc
    expected_indices = list(range(num_classes))
    if sorted(idx_to_class) != expected_indices:
        raise ValueError("idx_to_class must contain every contiguous class index exactly once")
    if any(not label.strip() for label in idx_to_class.values()):
        raise ValueError("idx_to_class contains an empty class label")
    if len(set(idx_to_class.values())) != num_classes:
        raise ValueError("idx_to_class contains duplicate class labels")
    raw_class_to_idx = payload.get("class_to_idx")
    if raw_class_to_idx is not None:
        if not isinstance(raw_class_to_idx, Mapping):
            raise ValueError("class_to_idx must be an object when present")
        class_to_idx = {str(label): int(index) for label, index in raw_class_to_idx.items()}
        expected_class_to_idx = {label: index for index, label in idx_to_class.items()}
        if class_to_idx != expected_class_to_idx:
            raise ValueError("class_to_idx and idx_to_class define different class orders")

    raw_preprocessing = _required_mapping(payload, "preprocessing")
    preprocessing = dict(raw_preprocessing)
    preprocessing_size = _positive_integer(preprocessing.get("image_size"), "preprocessing.image_size")
    if preprocessing_size != image_size:
        raise ValueError("Preprocessing image_size does not match the model input")
    resize_mode = _required_string(preprocessing, "resize_mode").lower()
    if resize_mode not in SUPPORTED_RESIZE_MODES:
        raise ValueError(f"Unsupported preprocessing resize_mode: {resize_mode}")
    try:
        crop_pct = float(preprocessing.get("crop_pct"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Preprocessing crop_pct must be numeric") from exc
    if not np.isfinite(crop_pct) or not 0.0 < crop_pct <= 1.0:
        raise ValueError("Preprocessing crop_pct must be in (0, 1]")
    interpolation = _required_string(preprocessing, "interpolation").lower()
    if interpolation not in SUPPORTED_INTERPOLATIONS:
        raise ValueError(f"Unsupported preprocessing interpolation: {interpolation}")
    mean = _three_channel_values(preprocessing, "mean")
    std = _three_channel_values(preprocessing, "std", positive=True)
    color_space = _required_string(preprocessing, "input_color_space").upper()
    if color_space != "RGB":
        raise ValueError("The image serving contract requires RGB preprocessing")
    preprocessing.update(
        {
            "image_size": preprocessing_size,
            "resize_mode": resize_mode,
            "crop_pct": crop_pct,
            "interpolation": interpolation,
            "mean": mean,
            "std": std,
            "input_color_space": color_space,
        }
    )

    calibration = _required_mapping(payload, "calibration")
    if "temperature" not in calibration:
        raise ValueError("Model calibration metadata has no temperature")
    try:
        temperature = float(calibration["temperature"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Model calibration temperature must be numeric") from exc
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Model calibration temperature must be finite and positive")

    onnx_metadata = _required_mapping(payload, "onnx")
    onnx_input_name = _required_string(onnx_metadata, "input_name")
    onnx_output_name = _required_string(onnx_metadata, "output_name")
    parity = _required_mapping(onnx_metadata, "parity")
    if parity.get("passed") is not True:
        raise ValueError("The production ONNX bundle does not have a passing parity result")

    return ModelMetadata(
        architecture=architecture,
        model_name=model_name,
        model_version=model_version,
        input_width=input_width,
        input_height=input_height,
        channels=channels,
        layout=layout,
        dtype=dtype,
        num_classes=num_classes,
        idx_to_class=idx_to_class,
        preprocessing=preprocessing,
        temperature=temperature,
        onnx_input_name=onnx_input_name,
        onnx_output_name=onnx_output_name,
    )


def _validate_release_metadata(release: ModelRelease, metadata: ModelMetadata) -> None:
    expected_input = (
        release.input_width,
        release.input_height,
        release.input_channels,
    )
    actual_input = (
        metadata.input_width,
        metadata.input_height,
        metadata.channels,
    )
    if metadata.architecture != release.architecture:
        raise ValueError("Model metadata architecture does not match the release manifest")
    if metadata.model_version != release.version:
        raise ValueError("Model metadata version does not match the release manifest")
    if actual_input != expected_input:
        raise ValueError("Model metadata input dimensions do not match the release manifest")
    if metadata.num_classes != release.class_count:
        raise ValueError("Model metadata class count does not match the release manifest")


def _validate_onnx_contract(session, metadata: ModelMetadata) -> None:
    inputs = {item.name: item for item in session.get_inputs()}
    outputs = {item.name: item for item in session.get_outputs()}
    if metadata.onnx_input_name not in inputs:
        raise ValueError(
            f"Metadata input name '{metadata.onnx_input_name}' is absent from the ONNX graph"
        )
    if metadata.onnx_output_name not in outputs:
        raise ValueError(
            f"Metadata output name '{metadata.onnx_output_name}' is absent from the ONNX graph"
        )
    model_input = inputs[metadata.onnx_input_name]
    if model_input.type != "tensor(float)":
        raise ValueError(f"ONNX input must be float32, found {model_input.type}")
    if len(model_input.shape) != 4:
        raise ValueError(f"ONNX input must have rank 4, found shape {model_input.shape}")
    expected_input = (metadata.channels, metadata.input_height, metadata.input_width)
    actual_input = tuple(model_input.shape[1:])
    if actual_input != expected_input:
        raise ValueError(
            f"ONNX input shape {model_input.shape} does not match metadata NCHW input "
            f"[batch, {metadata.channels}, {metadata.input_height}, {metadata.input_width}]"
        )
    model_output = outputs[metadata.onnx_output_name]
    if len(model_output.shape) != 2:
        raise ValueError(f"ONNX output must have rank 2, found shape {model_output.shape}")
    output_classes = model_output.shape[-1]
    if not isinstance(output_classes, int) or output_classes != metadata.num_classes:
        raise ValueError(
            f"ONNX output class dimension {output_classes} does not match metadata "
            f"class count {metadata.num_classes}"
        )


class ModelService:
    def __init__(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
    ) -> None:
        self.configured_model_path = model_path
        self.configured_metadata_path = metadata_path
        self.configured_manifest_path = manifest_path
        self._reset()

    def _reset(self) -> None:
        self.session = None
        self.metadata: ModelMetadata | None = None
        self.input_name: str | None = None
        self.output_name: str | None = None
        self.image_size: int | None = None
        self.preprocessing: dict = {}
        self.temperature: float | None = None
        self.idx_to_class: dict[int, str] = {}
        self.model_path: Path | None = None
        self.metadata_path: Path | None = None
        self.manifest_path: Path | None = None
        self.provider: str | None = None
        self.model_checksum: str | None = None
        self.load_error: str | None = None

    def load(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
    ) -> bool:
        configured_model = model_path or self.configured_model_path or settings.model_path
        configured_metadata = metadata_path or self.configured_metadata_path or settings.model_metadata_path
        resolved_model_path = resolve_project_path(configured_model)
        resolved_metadata_path = resolve_project_path(configured_metadata)
        configured_manifest = manifest_path or self.configured_manifest_path
        if configured_manifest is None:
            if model_path is not None or self.configured_model_path is not None:
                resolved_manifest_path = resolved_model_path.parent / "release.json"
            else:
                resolved_manifest_path = resolve_project_path(settings.model_release_manifest_path)
        else:
            resolved_manifest_path = resolve_project_path(configured_manifest)
        self._reset()
        self.model_path = resolved_model_path
        self.metadata_path = resolved_metadata_path
        self.manifest_path = resolved_manifest_path
        try:
            if not resolved_model_path.is_file():
                raise FileNotFoundError(f"ONNX model does not exist: {resolved_model_path}")
            release = load_release_manifest(resolved_manifest_path)
            expected_model_path = release.path_for(release.onnx).resolve()
            expected_metadata_path = release.path_for(release.metadata).resolve()
            if resolved_model_path != expected_model_path:
                raise ValueError(
                    f"Configured model path does not match release manifest: {expected_model_path}"
                )
            if resolved_metadata_path != expected_metadata_path:
                raise ValueError(
                    "Configured metadata path does not match release manifest: "
                    f"{expected_metadata_path}"
                )
            verify_supporting_assets(release)
            _, model_checksum = verify_asset(resolved_model_path, release.onnx)
            metadata = load_model_metadata(resolved_metadata_path)
            _validate_release_metadata(release, metadata)

            import onnxruntime as ort

            available = set(ort.get_available_providers())
            providers = [
                provider
                for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if provider in available
            ]
            if not providers:
                raise RuntimeError(
                    f"No supported ONNX Runtime provider is available; found {sorted(available)}"
                )
            session = ort.InferenceSession(str(resolved_model_path), providers=providers)
            _validate_onnx_contract(session, metadata)

            self.session = session
            self.metadata = metadata
            self.input_name = metadata.onnx_input_name
            self.output_name = metadata.onnx_output_name
            self.image_size = metadata.input_height
            self.preprocessing = metadata.preprocessing
            self.temperature = metadata.temperature
            self.idx_to_class = metadata.idx_to_class
            self.provider = session.get_providers()[0]
            self.model_checksum = model_checksum
            logger.info(
                "Loaded model architecture=%s version=%s input=%sx%s classes=%s "
                "provider=%s sha256=%s",
                metadata.architecture,
                metadata.model_version,
                metadata.input_width,
                metadata.input_height,
                metadata.num_classes,
                self.provider,
                self.model_checksum,
            )
            return True
        except Exception as exc:
            self.session = None
            self.metadata = None
            self.load_error = (
                f"Failed to load required ONNX model bundle: {exc}. "
                "Run 'python scripts/download_model.py' from the repository root to obtain "
                "or verify the configured release."
            )
            logger.error(self.load_error, exc_info=True)
            return False

    @property
    def loaded(self) -> bool:
        return self.session is not None and self.metadata is not None

    @property
    def mode(self) -> str:
        return "onnx" if self.loaded else "unavailable"

    @property
    def architecture(self) -> str | None:
        return self.metadata.architecture if self.metadata else None

    @property
    def model_name(self) -> str | None:
        return self.metadata.model_name if self.metadata else None

    @property
    def model_version(self) -> str | None:
        return self.metadata.model_version if self.metadata else None

    @property
    def input_size(self) -> dict[str, int] | None:
        return self.metadata.input_size if self.metadata else None

    def predict(self, image: Image.Image) -> dict:
        if not self.loaded or self.input_name is None or self.output_name is None:
            raise ModelUnavailableError(self.load_error or "ONNX model is not loaded")
        model_input = preprocess_for_onnx(image, int(self.image_size), self.preprocessing)
        logits = np.asarray(
            self.session.run([self.output_name], {self.input_name: model_input})[0],
            dtype=np.float64,
        )
        if logits.shape != (1, len(self.idx_to_class)) or not np.all(np.isfinite(logits)):
            raise RuntimeError(f"Unexpected ONNX output shape or values: {logits.shape}")
        probabilities = self._softmax(logits / float(self.temperature))[0]
        top_indices = probabilities.argsort()[-3:][::-1]
        top = [
            {
                "class_name": self.idx_to_class[int(index)],
                "confidence": float(probabilities[index]),
            }
            for index in top_indices
        ]
        return {
            "class_name": top[0]["class_name"],
            "confidence": top[0]["confidence"],
            "top_3_predictions": top,
            "mode": "onnx",
            "mock": False,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "input_size": self.input_size,
        }

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values, axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / exponent.sum(axis=1, keepdims=True)


model_service = ModelService()


def load_model() -> bool:
    return model_service.load()


def get_supported_classes() -> list[str]:
    if not model_service.loaded:
        raise ModelUnavailableError(model_service.load_error or "ONNX model is not loaded")
    return [model_service.idx_to_class[index] for index in sorted(model_service.idx_to_class)]
