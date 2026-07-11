from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import get_dataloaders, load_class_mapping
from src.models.baseline_cnn import BaselineCNN
from src.models.model_factory import build_model
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.config import load_config
from src.training.engine import build_class_weighted_loss, train_one_epoch, validate
from src.utils.seed import set_seed
from src.utils.visualization import plot_training_curves


def _build_architecture(name: str, num_classes: int, pretrained: bool):
    if name == "baseline_cnn":
        return BaselineCNN(num_classes)
    return build_model(name, num_classes, pretrained)


def train(config_path: str | None = None) -> Path:
    config = load_config(config_path)
    set_seed(config.seed)
    device = config.resolved_device()
    class_to_idx, idx_to_class = load_class_mapping(config.mapping_path)
    num_classes = len(class_to_idx)

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        data_dir=config.data_dir,
        mapping_path=config.mapping_path,
        image_size=config.image_size,
    )
    if len(train_loader.dataset) == 0:
        raise FileNotFoundError("No training images found. Run download_data.py and split_dataset.py first.")

    model = _build_architecture(config.architecture, num_classes, config.pretrained).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(config.num_epochs, 1)) if config.lr_scheduler == "cosine" else None
    criterion = (
        build_class_weighted_loss(train_loader.dataset, num_classes, device)
        if config.class_weighting
        else torch.nn.CrossEntropyLoss()
    )

    early_stopping = EarlyStopping(config.early_stopping_patience)
    checkpoint = ModelCheckpoint(config.checkpoint_dir)
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    json_path = log_dir / "training_log.json"
    history: list[dict] = []

    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        metadata = {
            "architecture": config.architecture,
            "num_classes": num_classes,
            "image_size": config.image_size,
            "class_to_idx": class_to_idx,
            "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
            "config": asdict(config),
        }
        checkpoint.step(model, val_loss, metadata)
        if early_stopping.step(val_loss):
            print(f"Early stopping after epoch {epoch}")
            break

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    plot_training_curves(history, "docs/training_curves.png")
    return checkpoint.best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crop disease classifier.")
    parser.add_argument("--config", default=None, help="YAML config path, e.g. configs/base.yaml")
    args = parser.parse_args()
    best_path = train(args.config)
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
