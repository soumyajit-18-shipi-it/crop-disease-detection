from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from src.data.transforms import get_transforms


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def image_to_tensor(image: bytes | Image.Image, image_size: int = 224):
    if isinstance(image, bytes):
        pil_image = Image.open(BytesIO(image)).convert("RGB")
    else:
        pil_image = image.convert("RGB")
    image_np = np.array(pil_image)
    transformed = get_transforms("inference", image_size)(image=image_np)["image"]
    return transformed.unsqueeze(0)


def preprocess_for_model(image: bytes | Image.Image, image_size: int = 224):
    return image_to_tensor(image, image_size)


def preprocess_for_onnx(image: bytes | Image.Image, image_size: int = 224) -> np.ndarray:
    tensor = image_to_tensor(image, image_size)
    return tensor.numpy().astype(np.float32)


def validate_image_bytes(image_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(image_bytes))
    image.verify()
    return Image.open(BytesIO(image_bytes)).convert("RGB")
