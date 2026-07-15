from __future__ import annotations

import timm
from torch import nn


SUPPORTED_ARCHITECTURES = {
    "efficientnet_b0": "efficientnet_b0",
    "resnet50": "resnet50",
    "mobilenetv3_large": "mobilenetv3_large_100",
}


def build_model(architecture: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a timm classifier for input tensors shaped (N, 3, 224, 224)."""
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture {architecture}. Choose one of {sorted(SUPPORTED_ARCHITECTURES)}")
    return timm.create_model(SUPPORTED_ARCHITECTURES[architecture], pretrained=pretrained, num_classes=num_classes)
