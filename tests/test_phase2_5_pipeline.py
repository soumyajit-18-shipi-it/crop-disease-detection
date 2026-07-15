from dataclasses import replace

import numpy as np
import torch

from src.evaluation.calibration import calibration_metrics, fit_temperature, softmax_probabilities
from src.inference.preprocess_input import preprocess_rgb_array
from src.training.benchmark import _select_best
from src.training.config import load_config
from src.training.engine import apply_batch_augmentation, build_class_weights


class _Dataset:
    targets = [0, 0, 0, 1]


def test_phase2_5_reuses_split_and_declares_exact_selection_weights():
    config = load_config("configs/training/phase2_5.yaml")
    assert config.data.split_manifest == "data/splits/phase1_split.json"
    assert config.data.use_model_preprocessing is True
    assert config.model.architectures == ["efficientnetv2_s", "convnext_tiny", "convnext_base"]
    assert "swin_tiny" in config.model.optional_architectures
    assert (
        config.selection.validation_macro_f1_weight,
        config.selection.calibration_weight,
        config.selection.inference_speed_weight,
        config.selection.model_size_weight,
        config.selection.memory_weight,
    ) == (0.40, 0.20, 0.15, 0.15, 0.10)


def test_metadata_driven_preprocessing_handles_non_square_input():
    image = np.full((40, 80, 3), 255, dtype=np.uint8)
    preprocessing = {
        "image_size": 32,
        "resize_mode": "shortest_center_crop",
        "crop_pct": 0.875,
        "interpolation": "bicubic",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    }
    output = preprocess_rgb_array(image, 32, preprocessing)
    assert output.shape == (1, 3, 32, 32)
    assert np.allclose(output, 1.0)


def test_temperature_scaling_never_returns_worse_validation_nll():
    logits = np.array([[4.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]])
    targets = np.array([0, 1, 1, 0, 0, 1])
    fitted = fit_temperature(logits, targets, max_iterations=30)
    before = calibration_metrics(targets, softmax_probabilities(logits))
    after = calibration_metrics(targets, softmax_probabilities(logits, fitted["temperature"]))
    assert fitted["temperature"] > 0
    assert after["negative_log_likelihood"] <= before["negative_log_likelihood"] + 1e-9


def test_mixup_and_cutmix_are_individually_configurable():
    base = load_config("configs/training/phase2_5.yaml").augmentation
    images = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8)
    labels = torch.arange(4)
    for augmentation, expected in (
        (replace(base, mixup_probability=1.0, cutmix_probability=0.0), "mixup"),
        (replace(base, mixup_probability=0.0, cutmix_probability=1.0), "cutmix"),
    ):
        mixed, first, second, lam, kind = apply_batch_augmentation(images, labels, augmentation)
        assert kind == expected
        assert mixed.shape == images.shape
        assert first.shape == second.shape == labels.shape
        assert 0.0 <= lam <= 1.0


def test_effective_number_weighting_upweights_the_rare_class():
    weights = build_class_weights(_Dataset(), 2, strategy="effective_num", effective_num_beta=0.99)
    assert weights is not None
    assert weights[1] > weights[0]


def test_selection_uses_memory_as_the_required_fifth_component():
    def metrics(memory):
        return {
            "validation": {"macro": {"f1": 0.9}, "calibration": {"expected_calibration_error": 0.05}},
            "onnx_parity": {"passed": True, "onnx_cpu_inference": {"images_per_second": 50.0}},
            "onnx_size_bytes": 100,
            "peak_gpu_memory_bytes": memory,
        }

    selected, scores = _select_best({"lower_memory": metrics(100), "higher_memory": metrics(200)})
    assert selected == "lower_memory"
    assert scores["lower_memory"] > scores["higher_memory"]
