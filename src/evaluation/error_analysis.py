from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix


def most_confused_pairs(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Rank unordered class pairs by their measured bidirectional mistakes."""
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    total_errors = int(matrix.sum() - np.trace(matrix))
    pairs: list[dict[str, Any]] = []
    for first in labels:
        for second in range(first + 1, len(class_names)):
            first_to_second = int(matrix[first, second])
            second_to_first = int(matrix[second, first])
            pair_errors = first_to_second + second_to_first
            if pair_errors == 0:
                continue
            first_support = int(matrix[first].sum())
            second_support = int(matrix[second].sum())
            pairs.append(
                {
                    "class_a": str(class_names[first]),
                    "class_b": str(class_names[second]),
                    "a_as_b": first_to_second,
                    "b_as_a": second_to_first,
                    "pair_errors": pair_errors,
                    "share_of_all_errors": (
                        float(pair_errors / total_errors) if total_errors else 0.0
                    ),
                    "a_as_b_rate": (
                        float(first_to_second / first_support) if first_support else 0.0
                    ),
                    "b_as_a_rate": (
                        float(second_to_first / second_support) if second_support else 0.0
                    ),
                }
            )
    pairs.sort(
        key=lambda item: (
            -item["pair_errors"],
            -item["share_of_all_errors"],
            item["class_a"],
            item["class_b"],
        )
    )
    return pairs[: max(int(limit), 0)]


def weakest_classes(
    per_class: Mapping[str, Mapping[str, float | int | None]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return the lowest-F1 classes with recall, precision, support, and AUC."""
    rows = [
        {
            "class_name": str(name),
            "precision": _optional_float(values.get("precision")),
            "recall": _optional_float(values.get("recall")),
            "f1": _optional_float(values.get("f1")),
            "support": int(values.get("support") or 0),
            "roc_auc": _optional_float(values.get("roc_auc")),
        }
        for name, values in per_class.items()
    ]
    rows.sort(
        key=lambda item: (
            float("inf") if item["f1"] is None else item["f1"],
            float("inf") if item["recall"] is None else item["recall"],
            item["support"],
            item["class_name"],
        )
    )
    return rows[: max(int(limit), 0)]


def support_f1_spearman(
    per_class: Mapping[str, Mapping[str, float | int | None]],
) -> float | None:
    """Measure whether low-support classes systematically have lower F1."""
    usable = [
        (float(values.get("support") or 0), _optional_float(values.get("f1")))
        for values in per_class.values()
    ]
    usable = [(support, f1) for support, f1 in usable if support > 0 and f1 is not None]
    if len(usable) < 3:
        return None
    support_ranks = _average_ranks(np.asarray([item[0] for item in usable], dtype=np.float64))
    f1_ranks = _average_ranks(np.asarray([item[1] for item in usable], dtype=np.float64))
    if np.std(support_ranks) == 0 or np.std(f1_ranks) == 0:
        return None
    return float(np.corrcoef(support_ranks, f1_ranks)[0, 1])


def learning_curve_evidence(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize observed fit dynamics without treating augmented accuracy as train accuracy."""
    if not history:
        return {
            "epochs_completed": 0,
            "overfitting_status": "not_assessable",
            "underfitting_status": "not_assessable",
            "reason": "No training history is available.",
        }

    rows = [
        {
            "epoch": int(row["epoch"]),
            "train_loss": float(row["train_loss"]),
            "val_loss": float(row["val_loss"]),
            "val_macro_f1": float(row["val_macro_f1"]),
        }
        for row in history
    ]
    best = max(rows, key=lambda row: (row["val_macro_f1"], -row["epoch"]))
    minimum_loss = min(rows, key=lambda row: (row["val_loss"], row["epoch"]))
    final = rows[-1]
    first = rows[0]
    final_f1_drop = float(best["val_macro_f1"] - final["val_macro_f1"])
    val_loss_increase = float(final["val_loss"] / max(minimum_loss["val_loss"], 1e-12) - 1.0)
    train_loss_decrease = float(
        1.0 - final["train_loss"] / max(first["train_loss"], 1e-12)
    )
    epochs_after_best = int(final["epoch"] - best["epoch"])

    overfit_signals = {
        "validation_macro_f1_drop_at_least_0_005": final_f1_drop >= 0.005,
        "validation_loss_increase_at_least_5_percent": val_loss_increase >= 0.05,
        "training_loss_decreased": train_loss_decrease > 0.0,
        "at_least_three_epochs_after_best": epochs_after_best >= 3,
    }
    decisive_overfit_signals = sum(
        bool(overfit_signals[name])
        for name in (
            "validation_macro_f1_drop_at_least_0_005",
            "validation_loss_increase_at_least_5_percent",
            "at_least_three_epochs_after_best",
        )
    )
    if decisive_overfit_signals == 3 and overfit_signals["training_loss_decreased"]:
        overfitting_status = "observed"
    elif decisive_overfit_signals >= 2:
        overfitting_status = "possible"
    else:
        overfitting_status = "not_observed"

    # MixUp/CutMix, class weighting, and label smoothing make recorded train
    # accuracy/loss deliberately non-comparable with clean validation metrics.
    # We therefore flag only curve-level evidence and do not invent a train-fit
    # threshold. A clean-train evaluation can be added to a selected model.
    still_improving = best["epoch"] == final["epoch"] and len(rows) >= 3
    low_validation_fit = best["val_macro_f1"] < 0.80
    if low_validation_fit and still_improving:
        underfitting_status = "possible_capacity_or_training_limit"
    elif low_validation_fit:
        underfitting_status = "possible_optimization_limit"
    else:
        underfitting_status = "not_observed_from_validation_curve"

    return {
        "epochs_completed": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_validation_macro_f1": float(best["val_macro_f1"]),
        "final_epoch": int(final["epoch"]),
        "final_validation_macro_f1": float(final["val_macro_f1"]),
        "validation_macro_f1_gain": float(best["val_macro_f1"] - first["val_macro_f1"]),
        "final_f1_drop_from_best": final_f1_drop,
        "minimum_validation_loss_epoch": int(minimum_loss["epoch"]),
        "minimum_validation_loss": float(minimum_loss["val_loss"]),
        "final_validation_loss": float(final["val_loss"]),
        "final_validation_loss_increase_fraction": val_loss_increase,
        "training_loss_decrease_fraction": train_loss_decrease,
        "overfitting_status": overfitting_status,
        "overfitting_signals": overfit_signals,
        "underfitting_status": underfitting_status,
        "train_metric_caveat": (
            "Recorded train metrics include MixUp/CutMix, class-weighted loss, "
            "and label smoothing and are not clean-train generalization metrics."
        ),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks
