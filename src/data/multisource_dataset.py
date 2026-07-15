from __future__ import annotations

import hashlib
import json
import numpy as np
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.registry import SampleRecord, load_registered_sources
from src.data.transforms import get_transforms
from src.training.config import TrainConfig


def _content_hash(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assign_unsplit(records: list[SampleRecord], config: TrainConfig) -> dict[str, str]:
    assignments: dict[str, str] = {}
    by_label: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)
    ratios = {
        "train": config.data.train_ratio,
        "val": config.data.val_ratio,
        "test": config.data.test_ratio,
    }
    rng = random.Random(config.runtime.seed)
    for label, label_records in sorted(by_label.items()):
        by_hash: dict[str, list[SampleRecord]] = defaultdict(list)
        for record in label_records:
            by_hash[_content_hash(record.path)].append(record)
        groups = list(by_hash.values())
        rng.shuffle(groups)
        targets = {name: len(label_records) * ratio for name, ratio in ratios.items()}
        counts = Counter()
        for group in sorted(groups, key=len, reverse=True):
            split = min(ratios, key=lambda name: counts[name] / max(targets[name], 1.0))
            for record in group:
                assignments[record.sample_id] = split
            counts[split] += len(group)
    return assignments


def build_split_manifest(config: TrainConfig, force: bool = False) -> dict[str, Any]:
    output = Path(config.data.split_manifest)
    if output.exists() and not force:
        manifest = json.loads(output.read_text(encoding="utf-8"))
        if manifest.get("seed") != config.runtime.seed:
            raise ValueError("Existing split manifest seed differs; pass force=True to rebuild")
        return manifest

    records, skipped = load_registered_sources(config.data.sources)
    unsplit = [record for record in records if record.preset_split is None]
    assignments = _assign_unsplit(unsplit, config)
    items = []
    for record in records:
        split = record.preset_split or assignments[record.sample_id]
        items.append({**record.__dict__, "split": split})
    classes = sorted({record.label for record in records})
    manifest = {
        "schema_version": "1.0", "seed": config.runtime.seed,
        "ratios": {"train": config.data.train_ratio, "val": config.data.val_ratio, "test": config.data.test_ratio},
        "class_to_idx": {label: index for index, label in enumerate(classes)},
        "idx_to_class": {str(index): label for index, label in enumerate(classes)},
        "skipped_optional_datasets": skipped, "items": items,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class MultiSourceDataset(Dataset):
    def __init__(
        self,
        manifest: dict[str, Any],
        split: str,
        image_size: int,
        augmentation=None,
        preprocessing: dict | None = None,
    ):
        self.split = split
        self.class_to_idx = manifest["class_to_idx"]
        self.samples = [
            (Path(item["path"]), self.class_to_idx[item["label"]], item["dataset"])
            for item in manifest["items"] if item["split"] == split
        ]
        self.targets = [sample[1] for sample in self.samples]
        self.transform = get_transforms(split, image_size, augmentation, preprocessing)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, _dataset = self.samples[index]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image=image)["image"], label


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _class_balanced_sampler(dataset: MultiSourceDataset, config: TrainConfig, generator: torch.Generator):
    counts = Counter(dataset.targets)
    source_weights = {source.name: float(source.weight) for source in config.data.sources}
    weights = [
        source_weights.get(dataset_name, 1.0) / counts[label]
        for _path, label, dataset_name in dataset.samples
    ]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=config.data.sampler_replacement,
        generator=generator,
    )


def create_dataloaders(
    config: TrainConfig,
    force_split: bool = False,
    manifest: dict[str, Any] | None = None,
    preprocessing: dict | None = None,
) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    manifest = manifest or build_split_manifest(config, force_split)
    loaders = {}
    image_size = int((preprocessing or {}).get("image_size") or config.data.image_size or 224)
    for split_index, split in enumerate(("train", "val", "test")):
        dataset = MultiSourceDataset(
            manifest,
            split,
            image_size,
            augmentation=config.augmentation if split == "train" else None,
            preprocessing=preprocessing,
        )
        if not len(dataset):
            raise ValueError(f"Persisted split {split!r} contains no samples")
        generator = torch.Generator().manual_seed(config.runtime.seed + split_index)
        sampler = (
            _class_balanced_sampler(dataset, config, generator)
            if split == "train" and config.data.class_balanced_sampling
            else None
        )
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": config.data.batch_size,
            "shuffle": split == "train" and sampler is None,
            "sampler": sampler,
            "num_workers": config.data.num_workers,
            "pin_memory": config.data.pin_memory and config.resolved_device().startswith("cuda"),
            "persistent_workers": (
                config.data.num_workers > 0
                and config.data.persistent_workers
                and not config.runtime.deterministic
            ),
            "worker_init_fn": _seed_worker,
            "generator": generator,
        }
        if config.data.num_workers > 0:
            loader_kwargs["prefetch_factor"] = config.data.prefetch_factor
        loader = DataLoader(**loader_kwargs)
        # Checkpoint/resume uses this state to preserve sample order exactly at
        # epoch boundaries. Persistent workers are disabled in deterministic mode.
        loader.crop_generator = generator
        loaders[split] = loader
    return loaders, manifest
