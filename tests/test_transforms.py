import numpy as np

from src.data.transforms import apply_transforms, get_eval_transforms


def test_eval_transforms_return_tensor_shape():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    tensor = apply_transforms(image, get_eval_transforms(image_size=64))
    assert tuple(tensor.shape) == (3, 64, 64)
