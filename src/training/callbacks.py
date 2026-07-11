from pathlib import Path

import torch


class CheckpointCallback:
    """Save model checkpoints when validation loss improves."""

    def __init__(self, checkpoint_dir: str = "models/checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")

    def maybe_save(self, model, val_loss: float, filename: str = "best_model.pth") -> bool:
        if val_loss >= self.best_loss:
            return False
        self.best_loss = val_loss
        torch.save(model.state_dict(), self.checkpoint_dir / filename)
        return True
