from src.models.baseline_cnn import BaselineCNN
from src.models.transfer_model import create_transfer_model


def build_model(name: str, num_classes: int, pretrained: bool = True):
    """Build a model by name."""
    if name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    return create_transfer_model(name, num_classes=num_classes, pretrained=pretrained)
