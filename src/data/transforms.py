from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size + 32, image_size + 32),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def val_transform(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_transforms(split: str, image_size: int = 224) -> A.Compose:
    if split == "train":
        return train_transform(image_size)
    if split in {"val", "test", "inference"}:
        return val_transform(image_size)
    raise ValueError(f"Unknown split: {split}")


def get_train_transforms(image_size: int = 224) -> A.Compose:
    return train_transform(image_size)


def get_eval_transforms(image_size: int = 224) -> A.Compose:
    return val_transform(image_size)


def apply_transforms(image, transforms: A.Compose):
    return transforms(image=image)["image"]
