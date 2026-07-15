from __future__ import annotations

import timm
from timm.data import resolve_model_data_config
from torch import nn


SUPPORTED_ARCHITECTURES = {
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "convnext_tiny": "convnext_tiny",
    "convnext_base": "convnext_base",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "resnet50": "resnet50",
    "mobilenetv3_large": "mobilenetv3_large_100",
}


def build_model(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float | None = None,
    drop_path_rate: float | None = None,
) -> nn.Module:
    """Build a supported timm classifier.

    Input resolution and normalization are deliberately resolved separately
    from the model's pretrained configuration instead of assuming 224 px and
    ImageNet mean/std for every backbone.
    """
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture {architecture}. Choose one of {sorted(SUPPORTED_ARCHITECTURES)}")
    kwargs = {}
    if dropout is not None:
        kwargs["drop_rate"] = float(dropout)
    if drop_path_rate is not None:
        kwargs["drop_path_rate"] = float(drop_path_rate)
    return timm.create_model(
        SUPPORTED_ARCHITECTURES[architecture],
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )


def resolve_preprocessing(
    model: nn.Module,
    requested_image_size: int | None,
    use_model_defaults: bool,
) -> dict:
    """Return the exact preprocessing contract to persist with the model."""
    if use_model_defaults:
        model_config = resolve_model_data_config(model)
        native_size = int(model_config["input_size"][-1])
        image_size = int(requested_image_size or native_size)
        crop_pct = float(model_config.get("crop_pct", 1.0))
        return {
            "image_size": image_size,
            "resize_mode": "shortest_center_crop",
            "crop_pct": crop_pct,
            "interpolation": str(model_config.get("interpolation", "bicubic")),
            "mean": [float(value) for value in model_config["mean"]],
            "std": [float(value) for value in model_config["std"]],
            "input_color_space": "RGB",
            "input_range": [0.0, 1.0],
            "source": "timm_pretrained_config",
        }
    image_size = int(requested_image_size or 224)
    return {
        "image_size": image_size,
        "resize_mode": "stretch",
        "crop_pct": 1.0,
        "interpolation": "linear",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_color_space": "RGB",
        "input_range": [0.0, 1.0],
        "source": "legacy_imagenet",
    }
