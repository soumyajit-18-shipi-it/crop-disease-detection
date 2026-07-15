from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    data_dir: str = "data/processed"
    mapping_path: str = "data/class_mapping.json"
    architecture: str = "efficientnet_b0"
    pretrained: bool = True
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    learning_rate: float = 0.0003
    num_epochs: int = 20
    optimizer: str = "AdamW"
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"
    early_stopping_patience: int = 5
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str = "models/checkpoints"
    log_dir: str = "docs/training_logs"
    class_weighting: bool = True

    def resolved_device(self) -> str:
        return detect_device() if self.device == "auto" else self.device


def load_config(path: str | Path | None = None) -> TrainConfig:
    config = TrainConfig()
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as file:
        data: dict[str, Any] = yaml.safe_load(file) or {}
    merged = asdict(config)
    merged.update(data)
    return TrainConfig(**merged)
