from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from src.training.config import DatasetSourceConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SampleRecord:
    path: str
    label: str
    dataset: str
    sample_id: str
    preset_split: str | None = None


Loader = Callable[[DatasetSourceConfig], list[SampleRecord]]
_REGISTRY: dict[str, Loader] = {}


def register_dataset(name: str) -> Callable[[Loader], Loader]:
    def decorator(loader: Loader) -> Loader:
        if name in _REGISTRY:
            raise ValueError(f"Dataset loader already registered: {name}")
        _REGISTRY[name] = loader
        return loader
    return decorator


def available_dataset_types() -> list[str]:
    return sorted(_REGISTRY)


def _sample(path: Path, label: str, dataset: str, split: str | None = None) -> SampleRecord:
    identity = hashlib.sha256(f"{dataset}\0{path.resolve().as_posix()}".encode()).hexdigest()
    return SampleRecord(path.as_posix(), label, dataset, identity, split)


@register_dataset("image_folder")
def load_image_folder(config: DatasetSourceConfig) -> list[SampleRecord]:
    root = Path(config.path)
    records = []
    if not root.exists():
        return records
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(_sample(path, class_dir.name, config.name))
    return records


@register_dataset("recursive_image_folder")
def load_recursive_image_folder(config: DatasetSourceConfig) -> list[SampleRecord]:
    """Discover class folders at any depth, including PlantDoc train/test layouts."""
    root = Path(config.path)
    records = []
    if not root.exists():
        return records
    for class_dir in sorted(path for path in root.rglob("*") if path.is_dir()):
        images = [
            path for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        records.extend(_sample(path, class_dir.name, config.name) for path in images)
    return records


@register_dataset("pre_split_image_folder")
def load_pre_split_image_folder(config: DatasetSourceConfig) -> list[SampleRecord]:
    root = Path(config.path)
    records = []
    if not root.exists():
        return records
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append(_sample(path, class_dir.name, config.name, split))
    return records


@register_dataset("validated_manifest")
def load_validated_manifest(config: DatasetSourceConfig) -> list[SampleRecord]:
    path = Path(config.path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for record in payload.get("records", []):
        validation = record.get("validation", {})
        # This hard gate cannot be relaxed through YAML.
        if validation.get("eligible_for_training") is not True:
            continue
        image_path = Path(str(record.get("image_path") or ""))
        label = str(validation.get("canonical_disease") or "").strip()
        if image_path.is_file() and label:
            records.append(_sample(image_path, label, config.name))
    return records


def load_registered_sources(sources: Iterable[DatasetSourceConfig]) -> tuple[list[SampleRecord], list[str]]:
    records: list[SampleRecord] = []
    skipped: list[str] = []
    for source in sources:
        if not source.enabled:
            continue
        if source.type not in _REGISTRY:
            raise ValueError(f"Unknown dataset type {source.type!r}; available: {available_dataset_types()}")
        loaded = _REGISTRY[source.type](source)
        if not loaded:
            if source.optional:
                skipped.append(source.name)
                continue
            raise FileNotFoundError(f"Required dataset {source.name!r} has no samples at {source.path}")
        records.extend(loaded)
    if not records:
        raise FileNotFoundError("No training samples were discovered from enabled dataset sources")
    return records, skipped
