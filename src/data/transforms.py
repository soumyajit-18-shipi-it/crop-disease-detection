from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Return augmentation pipeline for training leaf classifiers."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> A.Compose:
    """Return deterministic preprocessing for validation/test/inference."""
    return A.Compose([A.Resize(image_size, image_size), A.Normalize(), ToTensorV2()])


def apply_transforms(image: Any, transforms: A.Compose):
    """Apply albumentations transforms to a numpy RGB image."""
    return transforms(image=image)["image"]
