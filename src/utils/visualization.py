from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def show_image(image, title: str | None = None) -> None:
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def plot_training_curves(history: list[dict], output_path: str | Path = "docs/training_curves.png") -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
