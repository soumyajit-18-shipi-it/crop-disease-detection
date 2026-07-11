from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, class_names: list[str] | None = None) -> dict:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    labels = class_names or [str(i) for i in range(len(precision))]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro": {
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1": float(f1_macro),
        },
        "per_class": {
            labels[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(labels))
        },
        "report": classification_report(y_true, y_pred, target_names=labels, zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, class_names: list[str], output_path: str | Path = "docs/confusion_matrix.png") -> Path:
    matrix = confusion_matrix(y_true, y_pred)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.45), max(6, len(class_names) * 0.35)))
    im = ax.imshow(matrix, interpolation="nearest", cmap="YlGn")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def compute_accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def build_classification_report(y_true, y_pred, target_names=None) -> str:
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def build_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
