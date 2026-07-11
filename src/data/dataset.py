from __future__ import annotations

import json
from pathlib import Path

import cv2
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import get_transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class_mapping(mapping_path: str | Path = "data/class_mapping.json") -> tuple[dict[str, int], dict[int, str]]:
    with Path(mapping_path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if "class_to_idx" in payload:
        class_to_idx = {str(k): int(v) for k, v in payload["class_to_idx"].items()}
        idx_to_class = {int(k): str(v) for k, v in payload["idx_to_class"].items()}
    else:
        idx_to_class = {int(k): str(v) for k, v in payload.items()}
        class_to_idx = {v: k for k, v in idx_to_class.items()}
    return class_to_idx, idx_to_class


class CropDiseaseDataset(Dataset):
    """PyTorch dataset for data/processed/{split}/{class_name}/image.jpg."""

    def __init__(
        self,
        root_dir: str | Path = "data/processed",
        split: str = "train",
        mapping_path: str | Path = "data/class_mapping.json",
        image_size: int = 224,
        transform=None,
    ) -> None:
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.class_to_idx, self.idx_to_class = load_class_mapping(mapping_path)
        self.transform = transform or get_transforms(split, image_size)
        self.samples = self._find_samples()

    def _find_samples(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        if not self.root_dir.exists():
            return samples
        for class_name, label in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for path in class_dir.rglob("*"):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((path, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unreadable image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image=image)["image"]
        return image_tensor, label


LeafDiseaseDataset = CropDiseaseDataset


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 2,
    data_dir: str | Path = "data/processed",
    mapping_path: str | Path = "data/class_mapping.json",
    image_size: int = 224,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    loaders = []
    for split in ("train", "val", "test"):
        dataset = CropDiseaseDataset(data_dir, split, mapping_path, image_size)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=split == "train",
                num_workers=num_workers,
                pin_memory=True,
            )
        )
    return tuple(loaders)
