import pytest

torch = pytest.importorskip("torch")

from src.models.baseline_cnn import BaselineCNN


def test_baseline_cnn_forward_shape():
    model = BaselineCNN(num_classes=6)
    output = model(torch.zeros(2, 3, 64, 64))
    assert tuple(output.shape) == (2, 6)
