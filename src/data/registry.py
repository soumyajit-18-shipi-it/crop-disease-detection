from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from src.data.validate_field_survey_manifest import (
    TRAINING_MANIFEST_TYPE,
    PrivacyAuditError,
    validate_training_manifest,
)
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
    raise ValueError(
        "Detailed field-survey review manifests are not training-safe. "
        "Use dataset type 'field_survey_training_manifest' with "
        "data/manifests/field_survey/training_manifest.json."
    )


@register_dataset("field_survey_training_manifest")
def load_field_survey_training_manifest(config: DatasetSourceConfig) -> list[SampleRecord]:
    path = Path(config.path)
    if not path.exists():
        return []
    if path.name == "validated_manifest.json":
        raise ValueError(
            "Refusing to load validated_manifest.json as training data; "
            "use the sanitized field-survey training manifest."
        )
    try:
        validate_training_manifest(path, path_root=Path.cwd())
    except PrivacyAuditError as exc:
        raise ValueError(f"Invalid field-survey training manifest {path}: {exc}") from exc
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manifest_type") != TRAINING_MANIFEST_TYPE:
        raise ValueError(f"Invalid field-survey training manifest type in {path}")
    records = []
    for record in payload.get("records", []):
        image_path = Path(str(record["image_path"]))
        label = str(record["canonical_class"]).strip()
        identity = hashlib.sha256(
            f"{config.name}\0{record['record_id']}\0{record['image_sha256']}".encode()
        ).hexdigest()
        records.append(SampleRecord(image_path.as_posix(), label, config.name, identity))
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
