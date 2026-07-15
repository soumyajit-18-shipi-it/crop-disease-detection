from __future__ import annotations

from dataclasses import dataclass, field
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
class DatasetSourceConfig:
    name: str
    type: str
    path: str
    optional: bool = False
    enabled: bool = True
    weight: float = 1.0


@dataclass
class DataConfig:
    sources: list[DatasetSourceConfig] = field(default_factory=list)
    split_manifest: str = "data/splits/training_split.json"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    # Set to null together with use_model_preprocessing=true to use each
    # backbone's native pretrained resolution.
    image_size: int | None = 224
    use_model_preprocessing: bool = False
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    class_balanced_sampling: bool = False
    sampler_replacement: bool = True


@dataclass
class AugmentationConfig:
    enabled: bool = True
    random_resized_crop_scale: list[float] = field(default_factory=lambda: [0.72, 1.0])
    random_resized_crop_ratio: list[float] = field(default_factory=lambda: [0.80, 1.25])
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.2
    geometry_probability: float = 0.45
    shift_limit: float = 0.06
    scale_limit: float = 0.12
    rotate_limit: int = 25
    perspective_scale: float = 0.05
    color_probability: float = 0.60
    brightness_contrast_weight: float = 1.0
    clahe_weight: float = 0.35
    hue_saturation_value_weight: float = 0.55
    rgb_shift_weight: float = 0.25
    gamma_weight: float = 0.45
    blur_probability: float = 0.16
    motion_blur_weight: float = 0.35
    gaussian_blur_weight: float = 0.40
    defocus_weight: float = 0.25
    weather_probability: float = 0.10
    shadow_weight: float = 1.0
    fog_weight: float = 0.18
    # Artificial rain can hide small lesions, so it is supported but disabled
    # in the production preset unless field validation demonstrates a benefit.
    rain_weight: float = 0.0
    compression_probability: float = 0.14
    jpeg_quality_min: int = 65
    coarse_dropout_probability: float = 0.12
    coarse_dropout_holes: list[int] = field(default_factory=lambda: [1, 4])
    coarse_dropout_size: list[float] = field(default_factory=lambda: [0.03, 0.12])
    mixup_alpha: float = 0.2
    mixup_probability: float = 0.15
    cutmix_alpha: float = 1.0
    cutmix_probability: float = 0.15


@dataclass
class ModelConfig:
    architecture: str = "efficientnetv2_s"
    architectures: list[str] = field(
        default_factory=lambda: ["efficientnetv2_s", "convnext_tiny", "convnext_base"]
    )
    optional_architectures: list[str] = field(default_factory=list)
    pretrained: bool = True
    dropout: float = 0.2
    drop_path_rate: float = 0.1
    architecture_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class OptimConfig:
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.02
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    filter_bias_and_norm_from_weight_decay: bool = True
    layer_decay: float | None = None
    epochs: int = 30
    warmup_epochs: int = 3
    warmup_start_factor: float = 0.01
    min_learning_rate: float = 1e-6
    label_smoothing: float = 0.1
    class_weighting: bool = True
    class_weight_strategy: str = "effective_num"
    effective_num_beta: float = 0.9999
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "auto"
    mixed_precision: bool = True
    amp_dtype: str = "float16"
    deterministic: bool = True
    deterministic_warn_only: bool = True
    ema_enabled: bool = True
    ema_decay: float = 0.9999
    early_stopping_patience: int = 8
    early_stopping_monitor: str = "macro_f1"
    early_stopping_mode: str = "max"
    early_stopping_min_delta: float = 1e-4


@dataclass
class EvaluationConfig:
    calibration_bins: int = 15
    temperature_max_iterations: int = 75
    latency_warmup_iterations: int = 10
    latency_iterations: int = 50


@dataclass
class SelectionConfig:
    validation_macro_f1_weight: float = 0.40
    calibration_weight: float = 0.20
    inference_speed_weight: float = 0.15
    model_size_weight: float = 0.15
    memory_weight: float = 0.10
    require_all_candidates: bool = True


@dataclass
class OutputConfig:
    run_root: str = "artifacts/training"
    onnx_opset: int = 18
    parity_atol: float = 1e-4
    parity_rtol: float = 1e-3


@dataclass
class TrainConfig:
    experiment_name: str = "crop_disease_phase2_5"
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimConfig = field(default_factory=OptimConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def resolved_device(self) -> str:
        return detect_device() if self.runtime.device == "auto" else self.runtime.device


def _construct(data: dict[str, Any]) -> TrainConfig:
    sources = [DatasetSourceConfig(**item) for item in data.get("data", {}).get("sources", [])]
    data_values = {k: v for k, v in data.get("data", {}).items() if k != "sources"}
    return TrainConfig(
        experiment_name=data.get("experiment_name", "crop_disease_phase2_5"),
        data=DataConfig(sources=sources, **data_values),
        augmentation=AugmentationConfig(**data.get("augmentation", {})),
        model=ModelConfig(**data.get("model", {})),
        optimization=OptimConfig(**data.get("optimization", {})),
        runtime=RuntimeConfig(**data.get("runtime", {})),
        evaluation=EvaluationConfig(**data.get("evaluation", {})),
        selection=SelectionConfig(**data.get("selection", {})),
        output=OutputConfig(**data.get("output", {})),
    )


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def validate_config(config: TrainConfig) -> None:
    ratios = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio
    if abs(ratios - 1.0) > 1e-8:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    if not config.data.sources:
        raise ValueError("At least one dataset source must be configured")
    if any(source.weight <= 0 for source in config.data.sources):
        raise ValueError("Dataset source weights must be positive")
    if config.data.image_size is None and not config.data.use_model_preprocessing:
        raise ValueError("data.image_size may be null only when data.use_model_preprocessing is true")
    if config.data.image_size is not None and config.data.image_size < 32:
        raise ValueError("data.image_size must be at least 32")
    if config.data.batch_size < 1 or config.data.num_workers < 0:
        raise ValueError("data.batch_size must be positive and data.num_workers cannot be negative")
    if config.optimization.optimizer.lower() != "adamw":
        raise ValueError("Phase 2.5 currently supports optimizer=adamw")
    if config.optimization.learning_rate <= 0 or config.optimization.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay cannot be negative")
    if not 0 <= config.optimization.min_learning_rate <= config.optimization.learning_rate:
        raise ValueError("min_learning_rate must be between zero and learning_rate")
    if len(config.optimization.betas) != 2 or any(not 0.0 <= beta < 1.0 for beta in config.optimization.betas):
        raise ValueError("optimization.betas must contain two values in [0, 1)")
    if config.optimization.epochs < 1 or config.optimization.warmup_epochs < 0:
        raise ValueError("optimization.epochs must be positive and warmup_epochs cannot be negative")
    if not 0.0 < config.optimization.warmup_start_factor <= 1.0:
        raise ValueError("warmup_start_factor must be in (0, 1]")
    if config.optimization.layer_decay is not None and not 0.0 < config.optimization.layer_decay <= 1.0:
        raise ValueError("layer_decay must be in (0, 1] when configured")
    if config.optimization.gradient_accumulation_steps < 1:
        raise ValueError("optimization.gradient_accumulation_steps must be positive")
    if not 0.0 <= config.optimization.label_smoothing < 1.0:
        raise ValueError("optimization.label_smoothing must be in [0, 1)")
    if config.optimization.class_weight_strategy not in {"inverse_frequency", "effective_num"}:
        raise ValueError("class_weight_strategy must be inverse_frequency or effective_num")
    if not 0.0 < config.optimization.effective_num_beta < 1.0:
        raise ValueError("effective_num_beta must be in (0, 1)")
    if config.runtime.amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("runtime.amp_dtype must be float16 or bfloat16")
    if not 0.0 < config.runtime.ema_decay < 1.0:
        raise ValueError("runtime.ema_decay must be in (0, 1)")
    if config.runtime.early_stopping_monitor not in {"loss", "accuracy", "macro_f1"}:
        raise ValueError("runtime.early_stopping_monitor must be loss, accuracy, or macro_f1")
    if config.runtime.early_stopping_mode not in {"min", "max"}:
        raise ValueError("runtime.early_stopping_mode must be min or max")
    expected_mode = "min" if config.runtime.early_stopping_monitor == "loss" else "max"
    if config.runtime.early_stopping_mode != expected_mode:
        raise ValueError(
            f"early_stopping_mode must be {expected_mode} for {config.runtime.early_stopping_monitor}"
        )
    if config.runtime.early_stopping_patience < 1:
        raise ValueError("runtime.early_stopping_patience must be positive")
    if len(set(config.model.architectures)) != len(config.model.architectures):
        raise ValueError("model.architectures contains duplicates")
    if not 0.0 <= config.model.dropout < 1.0 or not 0.0 <= config.model.drop_path_rate < 1.0:
        raise ValueError("model dropout and drop_path_rate must be in [0, 1)")
    for name in (
        "horizontal_flip_probability", "vertical_flip_probability", "geometry_probability",
        "color_probability", "blur_probability", "weather_probability",
        "compression_probability", "coarse_dropout_probability", "mixup_probability",
        "cutmix_probability",
    ):
        _validate_probability(f"augmentation.{name}", float(getattr(config.augmentation, name)))
    if config.augmentation.mixup_probability + config.augmentation.cutmix_probability > 1.0:
        raise ValueError("MixUp and CutMix probabilities cannot sum to more than 1")
    if config.augmentation.mixup_alpha < 0 or config.augmentation.cutmix_alpha < 0:
        raise ValueError("MixUp and CutMix alpha values cannot be negative")
    for name in ("random_resized_crop_scale", "random_resized_crop_ratio", "coarse_dropout_size"):
        values = getattr(config.augmentation, name)
        if len(values) != 2 or values[0] <= 0 or values[0] > values[1]:
            raise ValueError(f"augmentation.{name} must be an ascending positive pair")
    if (
        len(config.augmentation.coarse_dropout_holes) != 2
        or config.augmentation.coarse_dropout_holes[0] < 1
        or config.augmentation.coarse_dropout_holes[0] > config.augmentation.coarse_dropout_holes[1]
    ):
        raise ValueError("augmentation.coarse_dropout_holes must be a positive pair")
    for name in (
        "brightness_contrast_weight", "clahe_weight", "hue_saturation_value_weight",
        "rgb_shift_weight", "gamma_weight", "motion_blur_weight", "gaussian_blur_weight",
        "defocus_weight", "shadow_weight", "fog_weight", "rain_weight",
    ):
        if getattr(config.augmentation, name) < 0:
            raise ValueError(f"augmentation.{name} cannot be negative")
    if not 1 <= config.augmentation.jpeg_quality_min <= 95:
        raise ValueError("augmentation.jpeg_quality_min must be between 1 and 95")
    if config.evaluation.calibration_bins < 2:
        raise ValueError("evaluation.calibration_bins must be at least 2")
    selection_total = sum(
        (
            config.selection.validation_macro_f1_weight,
            config.selection.calibration_weight,
            config.selection.inference_speed_weight,
            config.selection.model_size_weight,
            config.selection.memory_weight,
        )
    )
    if min(
        config.selection.validation_macro_f1_weight,
        config.selection.calibration_weight,
        config.selection.inference_speed_weight,
        config.selection.model_size_weight,
        config.selection.memory_weight,
    ) < 0:
        raise ValueError("Production-selection weights cannot be negative")
    if abs(selection_total - 1.0) > 1e-8:
        raise ValueError("Production-selection weights must sum to 1.0")


def config_from_dict(payload: dict[str, Any]) -> TrainConfig:
    config = _construct(payload)
    validate_config(config)
    return config


def load_config(path: str | Path = "configs/training/phase2_5.yaml") -> TrainConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    return config_from_dict(payload)
