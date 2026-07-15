from src.evaluation.error_analysis import (
    learning_curve_evidence,
    most_confused_pairs,
    support_f1_spearman,
    weakest_classes,
)
from src.evaluation.metrics import compute_accuracy


def test_compute_accuracy():
    assert compute_accuracy([0, 1, 1], [0, 0, 1]) == 2 / 3


def test_error_analysis_ranks_measured_pairs_and_weak_classes():
    pairs = most_confused_pairs(
        [0, 0, 1, 1, 2, 2],
        [1, 1, 0, 1, 2, 1],
        ["a", "b", "c"],
    )
    assert pairs[0]["class_a"] == "a"
    assert pairs[0]["class_b"] == "b"
    assert pairs[0]["pair_errors"] == 3
    assert pairs[0]["a_as_b"] == 2
    assert pairs[0]["b_as_a"] == 1

    per_class = {
        "a": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 100},
        "b": {"precision": 0.5, "recall": 0.4, "f1": 0.44, "support": 10},
        "c": {"precision": 0.7, "recall": 0.6, "f1": 0.65, "support": 50},
    }
    assert weakest_classes(per_class, 1)[0]["class_name"] == "b"
    assert support_f1_spearman(per_class) == 1.0


def test_learning_curve_evidence_flags_divergence_without_using_train_accuracy():
    history = [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 0.5, "val_macro_f1": 0.90},
        {"epoch": 2, "train_loss": 0.8, "val_loss": 0.51, "val_macro_f1": 0.89},
        {"epoch": 3, "train_loss": 0.6, "val_loss": 0.53, "val_macro_f1": 0.88},
        {"epoch": 4, "train_loss": 0.5, "val_loss": 0.56, "val_macro_f1": 0.87},
    ]

    evidence = learning_curve_evidence(history)

    assert evidence["overfitting_status"] == "observed"
    assert evidence["best_epoch"] == 1
    assert "MixUp/CutMix" in evidence["train_metric_caveat"]
