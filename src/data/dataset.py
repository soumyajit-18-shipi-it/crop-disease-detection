from pathlib import Path
from typing import Callable

from PIL import Image


class LeafDiseaseDataset:
    """Image dataset for folder layouts like train/class_name/image.jpg."""

    def __init__(self, root_dir: str | Path, transform: Callable | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()]) if self.root_dir.exists() else []
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = self._find_samples()

    def _find_samples(self) -> list[tuple[Path, int]]:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        samples: list[tuple[Path, int]] = []
        for class_name in self.classes:
            for image_path in (self.root_dir / class_name).rglob("*"):
                if image_path.suffix.lower() in extensions:
                    samples.append((image_path, self.class_to_idx[class_name]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
