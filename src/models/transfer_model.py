import timm
from torch import nn


def create_transfer_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a timm classifier with a crop disease output head."""
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
