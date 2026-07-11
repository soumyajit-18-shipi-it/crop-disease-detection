from __future__ import annotations

from pathlib import Path

import torch


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class ModelCheckpoint:
    def __init__(self, checkpoint_dir: str = "models/checkpoints", filename: str = "best_model.pth") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.best = float("inf")
        self.best_path = self.checkpoint_dir / filename

    def step(self, model, val_loss: float, metadata: dict) -> bool:
        if val_loss >= self.best:
            return False
        self.best = val_loss
        payload = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        }
        torch.save(payload, self.best_path)
        return True


CheckpointCallback = ModelCheckpoint
