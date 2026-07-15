from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_IMAGE_SIZE = 224

_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def load_image(image_path: str) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")


def image_to_rgb_array(image: bytes | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, bytes):
        return np.asarray(ImageOps.exif_transpose(Image.open(BytesIO(image))).convert("RGB"))
    if isinstance(image, Image.Image):
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image array shaped (H, W, 3).")
    return image.astype(np.uint8, copy=False)


def resolve_preprocessing_contract(
    image_size: int = DEFAULT_IMAGE_SIZE,
    preprocessing: dict | None = None,
) -> dict:
    """Resolve new bundle metadata while retaining legacy model behavior."""
    preprocessing = preprocessing or {}
    if "preprocessing" in preprocessing:
        preprocessing = preprocessing["preprocessing"] or {}
    return {
        "image_size": int(preprocessing.get("image_size", image_size)),
        "resize_mode": str(preprocessing.get("resize_mode", "stretch")),
        "crop_pct": float(preprocessing.get("crop_pct", 1.0)),
        "interpolation": str(preprocessing.get("interpolation", "linear")).lower(),
        "mean": np.asarray(preprocessing.get("mean", IMAGENET_MEAN), dtype=np.float32),
        "std": np.asarray(preprocessing.get("std", IMAGENET_STD), dtype=np.float32),
    }


def _resize_for_contract(image: np.ndarray, contract: dict) -> np.ndarray:
    image_size = int(contract["image_size"])
    interpolation = _INTERPOLATION.get(contract["interpolation"])
    if interpolation is None:
        raise ValueError(f"Unsupported interpolation: {contract['interpolation']}")
    if contract["resize_mode"] == "stretch":
        return cv2.resize(image, (image_size, image_size), interpolation=interpolation)
    if contract["resize_mode"] != "shortest_center_crop":
        raise ValueError(f"Unsupported resize mode: {contract['resize_mode']}")

    height, width = image.shape[:2]
    crop_pct = float(contract["crop_pct"])
    if not 0.0 < crop_pct <= 1.0:
        raise ValueError("crop_pct must be in (0, 1]")
    resize_shorter = max(image_size, int(round(image_size / crop_pct)))
    scale = resize_shorter / min(height, width)
    resized_width = max(image_size, int(round(width * scale)))
    resized_height = max(image_size, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    left = max((resized_width - image_size) // 2, 0)
    top = max((resized_height - image_size) // 2, 0)
    cropped = resized[top : top + image_size, left : left + image_size]
    if cropped.shape[:2] != (image_size, image_size):
        cropped = cv2.resize(cropped, (image_size, image_size), interpolation=interpolation)
    return cropped


def preprocess_rgb_array(
    image: bytes | Image.Image | np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    preprocessing: dict | None = None,
) -> np.ndarray:
    """Canonical eval/inference preprocessing.

    Input is RGB. Output is a float32 array shaped (1, 3, H, W), normalized
    with ImageNet statistics. Training/evaluation and ONNX serving both use
    this function for non-augmented preprocessing.
    """
    contract = resolve_preprocessing_contract(image_size, preprocessing)
    image_np = image_to_rgb_array(image)
    resized = _resize_for_contract(image_np, contract)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - contract["mean"]) / contract["std"]
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32, copy=False)


def image_to_tensor(
    image: bytes | Image.Image,
    image_size: int = 224,
    preprocessing: dict | None = None,
):
    import torch

    return torch.from_numpy(preprocess_rgb_array(image, image_size, preprocessing))


def preprocess_for_model(
    image: bytes | Image.Image,
    image_size: int = 224,
    preprocessing: dict | None = None,
):
    return image_to_tensor(image, image_size, preprocessing)


def preprocess_for_onnx(
    image: bytes | Image.Image,
    image_size: int = 224,
    preprocessing: dict | None = None,
) -> np.ndarray:
    return preprocess_rgb_array(image, image_size, preprocessing)


def validate_image_bytes(image_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(image_bytes))
    image.verify()
    return ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
