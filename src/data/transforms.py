from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2

from src.inference.preprocess_input import preprocess_rgb_array, resolve_preprocessing_contract


_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def _default_augmentation() -> SimpleNamespace:
    # Keep direct transform calls backward compatible with the historical
    # light augmentation policy; production YAML supplies the full policy.
    return SimpleNamespace(
        enabled=True,
        random_resized_crop_scale=[0.80, 1.0],
        random_resized_crop_ratio=[0.85, 1.18],
        horizontal_flip_probability=0.5,
        vertical_flip_probability=0.2,
        geometry_probability=0.35,
        shift_limit=0.05,
        scale_limit=0.10,
        rotate_limit=20,
        perspective_scale=0.04,
        color_probability=0.40,
        brightness_contrast_weight=1.0,
        clahe_weight=0.25,
        hue_saturation_value_weight=0.35,
        rgb_shift_weight=0.15,
        gamma_weight=0.30,
        blur_probability=0.10,
        motion_blur_weight=0.30,
        gaussian_blur_weight=0.45,
        defocus_weight=0.25,
        weather_probability=0.05,
        shadow_weight=1.0,
        fog_weight=0.10,
        rain_weight=0.0,
        compression_probability=0.08,
        jpeg_quality_min=70,
        coarse_dropout_probability=0.08,
        coarse_dropout_holes=[1, 3],
        coarse_dropout_size=[0.03, 0.10],
    )


def _weighted_one_of(items: list[tuple[A.BasicTransform, float]], probability: float):
    transforms = []
    for transform, weight in items:
        if weight > 0:
            transform.p = float(weight)
            transforms.append(transform)
    if not transforms or probability <= 0:
        return None
    return A.OneOf(transforms, p=float(probability))


class CanonicalEvalTransform:
    """Use the same NumPy preprocessing code as PyTorch and ONNX serving."""

    def __init__(self, image_size: int = 224, preprocessing: dict | None = None) -> None:
        self.image_size = image_size
        self.preprocessing = preprocessing

    def __call__(self, image):
        tensor = torch.from_numpy(
            preprocess_rgb_array(image, self.image_size, self.preprocessing)
        ).squeeze(0)
        return {"image": tensor}


def train_transform(
    image_size: int = 224,
    augmentation=None,
    preprocessing: dict | None = None,
) -> A.Compose:
    augmentation = augmentation or _default_augmentation()
    contract = resolve_preprocessing_contract(image_size, preprocessing)
    interpolation_name = contract["interpolation"]
    if interpolation_name not in _INTERPOLATION:
        raise ValueError(f"Unsupported interpolation: {interpolation_name}")
    interpolation = _INTERPOLATION[interpolation_name]
    mean = tuple(float(value) for value in contract["mean"])
    std = tuple(float(value) for value in contract["std"])

    if not augmentation.enabled:
        return CanonicalEvalTransform(image_size, preprocessing)

    transforms: list[A.BasicTransform] = [
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=tuple(float(v) for v in augmentation.random_resized_crop_scale),
            ratio=tuple(float(v) for v in augmentation.random_resized_crop_ratio),
            interpolation=interpolation,
            p=1.0,
        ),
        A.HorizontalFlip(p=float(augmentation.horizontal_flip_probability)),
        A.VerticalFlip(p=float(augmentation.vertical_flip_probability)),
    ]

    geometry = _weighted_one_of(
        [
            (
                A.Affine(
                    translate_percent=(-float(augmentation.shift_limit), float(augmentation.shift_limit)),
                    scale=(1.0 - float(augmentation.scale_limit), 1.0 + float(augmentation.scale_limit)),
                    rotate=(-int(augmentation.rotate_limit), int(augmentation.rotate_limit)),
                    interpolation=interpolation,
                    border_mode=cv2.BORDER_REFLECT_101,
                    balanced_scale=True,
                ),
                0.72,
            ),
            (
                A.Perspective(
                    scale=(0.02, float(augmentation.perspective_scale)),
                    keep_size=True,
                    interpolation=interpolation,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                0.28,
            ),
        ],
        float(augmentation.geometry_probability),
    )
    if geometry:
        transforms.append(geometry)

    color = _weighted_one_of(
        [
            (A.RandomBrightnessContrast(0.18, 0.18), augmentation.brightness_contrast_weight),
            (A.CLAHE(clip_limit=(1.0, 2.5), tile_grid_size=(8, 8)), augmentation.clahe_weight),
            (
                A.HueSaturationValue(
                    hue_shift_limit=8,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                ),
                augmentation.hue_saturation_value_weight,
            ),
            (A.RGBShift(10, 10, 10), augmentation.rgb_shift_weight),
            (A.RandomGamma(gamma_limit=(85, 115)), augmentation.gamma_weight),
        ],
        float(augmentation.color_probability),
    )
    if color:
        transforms.append(color)

    blur = _weighted_one_of(
        [
            (A.MotionBlur(blur_limit=(3, 7)), augmentation.motion_blur_weight),
            (A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.2, 1.4)), augmentation.gaussian_blur_weight),
            (A.Defocus(radius=(2, 4), alias_blur=(0.1, 0.35)), augmentation.defocus_weight),
        ],
        float(augmentation.blur_probability),
    )
    if blur:
        transforms.append(blur)

    weather = _weighted_one_of(
        [
            (
                A.RandomShadow(
                    shadow_roi=(0.0, 0.0, 1.0, 1.0),
                    num_shadows_limit=(1, 2),
                    shadow_dimension=5,
                    shadow_intensity_range=(0.55, 0.80),
                ),
                augmentation.shadow_weight,
            ),
            (A.RandomFog(alpha_coef=0.04, fog_coef_range=(0.10, 0.30)), augmentation.fog_weight),
            (
                A.RandomRain(
                    slant_range=(-6, 6),
                    drop_length=10,
                    blur_value=3,
                    brightness_coefficient=0.92,
                    rain_type="drizzle",
                ),
                augmentation.rain_weight,
            ),
        ],
        float(augmentation.weather_probability),
    )
    if weather:
        transforms.append(weather)

    if augmentation.compression_probability > 0:
        transforms.append(
            A.ImageCompression(
                compression_type="jpeg",
                quality_range=(int(augmentation.jpeg_quality_min), 95),
                p=float(augmentation.compression_probability),
            )
        )
    if augmentation.coarse_dropout_probability > 0:
        transforms.append(
            A.CoarseDropout(
                num_holes_range=tuple(int(v) for v in augmentation.coarse_dropout_holes),
                hole_height_range=tuple(float(v) for v in augmentation.coarse_dropout_size),
                hole_width_range=tuple(float(v) for v in augmentation.coarse_dropout_size),
                fill="random_uniform",
                p=float(augmentation.coarse_dropout_probability),
            )
        )
    transforms.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
    return A.Compose(transforms)


def val_transform(
    image_size: int = 224,
    preprocessing: dict | None = None,
) -> CanonicalEvalTransform:
    return CanonicalEvalTransform(image_size, preprocessing)


def get_transforms(
    split: str,
    image_size: int = 224,
    augmentation=None,
    preprocessing: dict | None = None,
):
    if split == "train":
        return train_transform(image_size, augmentation, preprocessing)
    if split in {"val", "test", "inference"}:
        return val_transform(image_size, preprocessing)
    raise ValueError(f"Unknown split: {split}")


def get_train_transforms(image_size: int = 224) -> A.Compose:
    return train_transform(image_size)


def get_eval_transforms(image_size: int = 224) -> CanonicalEvalTransform:
    return val_transform(image_size)


def apply_transforms(image, transforms):
    return transforms(image=image)["image"]
