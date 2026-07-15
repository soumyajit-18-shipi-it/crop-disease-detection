from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, cohen_kappa_score,
    confusion_matrix, matthews_corrcoef, precision_recall_fscore_support,
    roc_auc_score,
)

from src.evaluation.calibration import calibration_metrics


def compute_metrics(
    y_true,
    y_pred,
    class_names: list[str] | None = None,
    probabilities=None,
    calibration_bins_count: int = 15,
) -> dict:
    label_indices = list(range(len(class_names))) if class_names else None
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_indices, average="macro", zero_division=0
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_indices, average=None, zero_division=0
    )
    labels = class_names or [str(i) for i in range(len(precision))]
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=label_indices, average="weighted", zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro": {
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1": float(f1_macro),
        },
        "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]), "f1": float(weighted[2])},
        "per_class": {
            labels[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(labels))
        },
        "report": classification_report(y_true, y_pred, labels=label_indices, target_names=labels, zero_division=0),
    }
    if probabilities is not None and len(probabilities):
        scores = np.asarray(probabilities, dtype=np.float64)
        result["calibration"] = calibration_metrics(
            y_true, scores, bins=calibration_bins_count
        )
        try:
            result["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, scores, labels=label_indices, multi_class="ovr", average="macro"))
            result["roc_auc_ovr_weighted"] = float(
                roc_auc_score(y_true, scores, labels=label_indices, multi_class="ovr", average="weighted")
            )
        except ValueError:
            result["roc_auc_ovr_macro"] = None
            result["roc_auc_ovr_weighted"] = None
        targets = np.asarray(y_true)
        for index, label in enumerate(labels):
            binary_targets = targets == index
            try:
                result["per_class"][label]["roc_auc"] = float(
                    roc_auc_score(binary_targets, scores[:, index])
                )
            except ValueError:
                result["per_class"][label]["roc_auc"] = None
    return result


def plot_confusion_matrix(y_true, y_pred, class_names: list[str], output_path: str | Path = "docs/confusion_matrix.png") -> Path:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
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
