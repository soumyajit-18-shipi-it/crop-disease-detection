from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import albumentations
import onnxruntime
import sklearn
import timm
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from timm.optim import create_optimizer_v2
from torch.optim.lr_scheduler import LambdaLR

from src.data.multisource_dataset import build_split_manifest, create_dataloaders
from src.evaluation.calibration import (
    calibration_metrics,
    fit_temperature,
    plot_reliability_diagram,
    softmax_probabilities,
)
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix
from src.models.model_factory import build_model, resolve_preprocessing
from src.training.config import TrainConfig, load_config, validate_config
from src.training.engine import ModelEMA, build_class_weighted_loss, run_epoch
from src.training.export_onnx import export_and_verify_onnx
from src.utils.seed import set_seed


CHECKPOINT_SCHEMA_VERSION = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _scheduler(
    optimizer,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int,
    warmup_start_factor: float = 0.01,
    min_learning_rate: float = 0.0,
):
    total_steps = max(int(steps_per_epoch) * int(epochs), 1)
    warmup_steps = min(int(steps_per_epoch) * int(warmup_epochs), max(total_steps - 1, 0))
    base_lr = float(optimizer.param_groups[0]["lr"])
    minimum_factor = min(max(float(min_learning_rate) / max(base_lr, 1e-12), 0.0), 1.0)

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            progress = step / max(warmup_steps, 1)
            return warmup_start_factor + (1.0 - warmup_start_factor) * progress
        cosine_steps = max(total_steps - warmup_steps - 1, 1)
        progress = min(max((step - warmup_steps) / cosine_steps, 0.0), 1.0)
        return minimum_factor + 0.5 * (1.0 - minimum_factor) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _cpu_rng_tensor(value: torch.Tensor) -> torch.Tensor:
    """Return an RNG state in the device/dtype required by PyTorch RNG APIs.

    Resume checkpoints are loaded with ``map_location=device`` so optimizer
    tensors land on the training device. That also moves RNG byte tensors to
    CUDA, while ``set_rng_state`` and ``Generator.set_state`` require CPU
    ByteTensors.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError("Serialized RNG state must be a torch.Tensor")
    return value.detach().to(device="cpu", dtype=torch.uint8)


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_rng_tensor(state["torch"]))
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(
            [_cpu_rng_tensor(device_state) for device_state in state["cuda"]]
        )


def _training_signature(config: TrainConfig, architecture: str, split_hash: str, preprocessing: dict) -> str:
    payload = {
        "architecture": architecture,
        "split_manifest_sha256": split_hash,
        "preprocessing": preprocessing,
        "data": {
            "batch_size": config.data.batch_size,
            "class_balanced_sampling": config.data.class_balanced_sampling,
        },
        "model": {
            "dropout": config.model.dropout,
            "drop_path_rate": config.model.drop_path_rate,
        },
        "optimization": asdict(config.optimization),
        "augmentation": asdict(config.augmentation),
        "ema": {
            "enabled": config.runtime.ema_enabled,
            "decay": config.runtime.ema_decay,
        },
        "software": {"torch": torch.__version__, "timm": timm.__version__},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _save_resume_checkpoint(
    path: Path,
    model,
    ema,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    metadata: dict,
    epoch_metrics: dict,
    history: list[dict],
    best_score: float,
    best_epoch: int,
    bad_epochs: int,
    training_seconds: float,
    peak_gpu_memory_bytes: int | None,
    train_generator_state,
) -> None:
    _atomic_torch_save(
        path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": "resume",
            "training_model_state_dict": model.state_dict(),
            "ema_state_dict": ema.module.state_dict() if ema is not None else None,
            "ema_num_updates": ema.num_updates if ema is not None else 0,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "metadata": metadata,
            "metrics": epoch_metrics,
            "history": history,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "training_seconds": training_seconds,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "rng_state": _rng_state(),
            "train_generator_state": train_generator_state,
        },
    )


def _save_inference_checkpoint(path: Path, model, metadata: dict, epoch: int, metrics: dict) -> None:
    _atomic_torch_save(
        path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": "inference",
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "metadata": metadata,
            "metrics": metrics,
        },
    )


def _latency(
    model,
    device: str,
    image_size: int,
    warmup_iterations: int,
    iterations: int,
) -> dict:
    model.eval().to(device)
    sample = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(sample)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(sample)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    median_ms = statistics.median(timings)
    return {
        "device": device,
        "median_latency_ms": median_ms,
        "mean_latency_ms": statistics.mean(timings),
        "p90_latency_ms": float(np.percentile(timings, 90)),
        "p95_latency_ms": float(np.percentile(timings, 95)),
        "images_per_second": 1000.0 / median_ms,
        "batch_size": 1,
        "warmup_iterations": warmup_iterations,
        "iterations": iterations,
    }


def _apply_architecture_override(config: TrainConfig, architecture: str) -> None:
    override = config.model.architecture_overrides.get(architecture, {})
    destinations = {
        "batch_size": config.data,
        "image_size": config.data,
        "num_workers": config.data,
        "gradient_accumulation_steps": config.optimization,
        "learning_rate": config.optimization,
        "weight_decay": config.optimization,
        "layer_decay": config.optimization,
        "dropout": config.model,
        "drop_path_rate": config.model,
    }
    unknown = sorted(set(override) - set(destinations))
    if unknown:
        raise ValueError(f"Unsupported architecture override keys for {architecture}: {unknown}")
    for key, value in override.items():
        setattr(destinations[key], key, value)


def _monitor_value(config: TrainConfig, validation_result: dict) -> float:
    monitor = config.runtime.early_stopping_monitor
    if monitor == "loss":
        return float(validation_result["loss"])
    if monitor == "accuracy":
        return float(validation_result["accuracy"])
    return float(
        f1_score(
            validation_result["targets"],
            validation_result["predictions"],
            labels=list(range(len(validation_result["logits"][0]))),
            average="macro",
            zero_division=0,
        )
    )


def _is_improved(config: TrainConfig, value: float, best: float) -> bool:
    delta = config.runtime.early_stopping_min_delta
    if config.runtime.early_stopping_mode == "max":
        return value > best + delta
    return value < best - delta


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@contextmanager
def _run_lock(run_dir: Path):
    lock_path = run_dir / ".train.lock"
    run_dir.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
            owner = int(lock_payload.get("pid", -1))
        except (ValueError, OSError, json.JSONDecodeError):
            owner = -1
        if _pid_is_running(owner):
            raise RuntimeError(f"Training run is already active in {run_dir} (PID {owner})")
        lock_path.unlink(missing_ok=True)
    with lock_path.open("x", encoding="utf-8") as file:
        json.dump({"pid": os.getpid(), "created_at": _utc_now()}, file)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _write_history(run_dir: Path, history: list[dict]) -> None:
    _atomic_json(run_dir / "history.json", history)
    if not history:
        return
    fieldnames = list(dict.fromkeys(key for row in history for key in row))
    for filename in ("history.csv", "training_history.csv"):
        temporary = run_dir / f"{filename}.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        os.replace(temporary, run_dir / filename)


def _validate_resume_checkpoint(state: dict, metadata: dict) -> None:
    saved = state.get("metadata", {})
    for key in ("architecture", "num_classes", "split_manifest_sha256", "training_signature"):
        if saved.get(key) != metadata.get(key):
            raise ValueError(
                f"Cannot resume: checkpoint {key}={saved.get(key)!r} does not match current {metadata.get(key)!r}"
            )


def _finalize_run(
    config: TrainConfig,
    architecture: str,
    run_dir: Path,
    model,
    loaders,
    criterion,
    metadata: dict,
    history: list[dict],
    best_path: Path,
    best_epoch: int,
    training_seconds: float,
    peak_gpu_memory_bytes: int | None,
    device: str,
) -> Path:
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    validation_result = run_epoch(model, loaders["val"], criterion, device)
    test_result = run_epoch(model, loaders["test"], criterion, device)
    class_names = [metadata["idx_to_class"][str(index)] for index in range(metadata["num_classes"])]

    temperature_fit = fit_temperature(
        validation_result["logits"],
        validation_result["targets"],
        config.evaluation.temperature_max_iterations,
    )
    temperature = float(temperature_fit["temperature"])
    validation_raw_probabilities = softmax_probabilities(validation_result["logits"])
    validation_probabilities = softmax_probabilities(validation_result["logits"], temperature)
    test_raw_probabilities = softmax_probabilities(test_result["logits"])
    test_probabilities = softmax_probabilities(test_result["logits"], temperature)
    calibration_report = {
        **temperature_fit,
        "fitted_on": "validation",
        "bins": config.evaluation.calibration_bins,
        "validation": {
            "before": calibration_metrics(
                validation_result["targets"], validation_raw_probabilities, config.evaluation.calibration_bins
            ),
            "after": calibration_metrics(
                validation_result["targets"], validation_probabilities, config.evaluation.calibration_bins
            ),
        },
        "test": {
            "before": calibration_metrics(
                test_result["targets"], test_raw_probabilities, config.evaluation.calibration_bins
            ),
            "after": calibration_metrics(
                test_result["targets"], test_probabilities, config.evaluation.calibration_bins
            ),
        },
    }
    metadata["calibration"] = {
        "method": "temperature_scaling",
        "temperature": temperature,
        "fitted_on": "validation",
        "validation_ece_before": calibration_report["validation"]["before"]["expected_calibration_error"],
        "validation_ece_after": calibration_report["validation"]["after"]["expected_calibration_error"],
    }
    metadata.update(
        {
            "candidate_version": f"phase2_5_{architecture}_v1",
            "input_dimensions": [None, 3, metadata["image_size"], metadata["image_size"]],
            "channel_count": 3,
            "ordered_classes": class_names,
            "source_checkpoint_epoch": int(best_epoch),
        }
    )

    metrics = compute_metrics(
        test_result["targets"],
        test_result["predictions"],
        class_names,
        test_probabilities,
        config.evaluation.calibration_bins,
    )
    validation_metrics = compute_metrics(
        validation_result["targets"],
        validation_result["predictions"],
        class_names,
        validation_probabilities,
        config.evaluation.calibration_bins,
    )
    metrics.update(
        {
            "schema_version": "2.5",
            "architecture": architecture,
            "image_size": metadata["image_size"],
            "test_loss": test_result["loss"],
            "validation": validation_metrics,
            "calibration_summary": {
                "temperature": temperature,
                "validation_ece_before": calibration_report["validation"]["before"]["expected_calibration_error"],
                "validation_ece_after": calibration_report["validation"]["after"]["expected_calibration_error"],
                "test_ece_before": calibration_report["test"]["before"]["expected_calibration_error"],
                "test_ece_after": calibration_report["test"]["after"]["expected_calibration_error"],
            },
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "parameter_size_bytes": sum(
                parameter.numel() * parameter.element_size() for parameter in model.parameters()
            ),
            "training_time_seconds": training_seconds,
            "training_batch_size": config.data.batch_size,
            "gradient_accumulation_steps": config.optimization.gradient_accumulation_steps,
            "effective_batch_size": (
                config.data.batch_size * config.optimization.gradient_accumulation_steps
            ),
            "best_epoch": best_epoch,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "split_manifest_sha256": metadata["split_manifest_sha256"],
            "device": device,
        }
    )

    warmups = config.evaluation.latency_warmup_iterations
    iterations = config.evaluation.latency_iterations
    metrics["device_inference"] = _latency(
        model, device, metadata["image_size"], warmups, iterations
    )
    metrics["cpu_inference"] = _latency(
        model, "cpu", metadata["image_size"], warmups, iterations
    )
    _atomic_json(run_dir / "calibration.json", calibration_report)
    np.savez_compressed(
        run_dir / "evaluation_logits.npz",
        validation_logits=np.asarray(validation_result["logits"], dtype=np.float32),
        validation_targets=np.asarray(validation_result["targets"], dtype=np.int64),
        test_logits=np.asarray(test_result["logits"], dtype=np.float32),
        test_targets=np.asarray(test_result["targets"], dtype=np.int64),
        temperature=np.asarray([temperature], dtype=np.float64),
    )
    _atomic_json(
        run_dir / "classification_report.json",
        classification_report(
            test_result["targets"],
            test_result["predictions"],
            labels=list(range(metadata["num_classes"])),
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
    )
    (run_dir / "classification_report.txt").write_text(
        classification_report(
            test_result["targets"],
            test_result["predictions"],
            labels=list(range(metadata["num_classes"])),
            target_names=class_names,
            zero_division=0,
        ),
        encoding="utf-8",
    )
    plot_confusion_matrix(
        test_result["targets"], test_result["predictions"], class_names, run_dir / "confusion_matrix.png"
    )
    _atomic_json(
        run_dir / "confusion_matrix.json",
        {
            "class_names": class_names,
            "matrix": confusion_matrix(
                test_result["targets"],
                test_result["predictions"],
                labels=list(range(metadata["num_classes"])),
            ).tolist(),
        },
    )
    confidence_rows = []
    raw_confidence = np.max(test_raw_probabilities, axis=1)
    calibrated_confidence = np.max(test_probabilities, axis=1)
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        confidence_rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "raw_count": int(((raw_confidence >= lower) & (raw_confidence <= upper)).sum()),
                "calibrated_count": int(
                    ((calibrated_confidence >= lower) & (calibrated_confidence <= upper)).sum()
                ),
            }
        )
    _atomic_json(
        run_dir / "confidence_distribution.json",
        {
            "samples": len(test_result["targets"]),
            "raw": {
                "minimum": float(raw_confidence.min()),
                "median": float(np.median(raw_confidence)),
                "mean": float(raw_confidence.mean()),
                "maximum": float(raw_confidence.max()),
            },
            "calibrated": {
                "minimum": float(calibrated_confidence.min()),
                "median": float(np.median(calibrated_confidence)),
                "mean": float(calibrated_confidence.mean()),
                "maximum": float(calibrated_confidence.max()),
            },
            "histogram": confidence_rows,
        },
    )
    test_samples = getattr(loaders["test"].dataset, "samples", [])
    misclassified = []
    for index, (target, prediction) in enumerate(
        zip(test_result["targets"], test_result["predictions"])
    ):
        if target == prediction:
            continue
        probabilities = test_probabilities[index]
        top3 = np.argsort(probabilities)[::-1][:3]
        path = str(test_samples[index][0]) if index < len(test_samples) else None
        misclassified.append(
            {
                "index": index,
                "path": path,
                "true_index": int(target),
                "true_class": class_names[target],
                "predicted_index": int(prediction),
                "predicted_class": class_names[prediction],
                "confidence": float(probabilities[prediction]),
                "top3": [
                    {"class": class_names[int(item)], "probability": float(probabilities[item])}
                    for item in top3
                ],
            }
        )
    _atomic_json(run_dir / "misclassified_images.json", misclassified)
    plot_reliability_diagram(
        test_result["targets"],
        test_result["logits"],
        temperature,
        run_dir / "reliability_diagram.png",
        config.evaluation.calibration_bins,
    )
    _write_history(run_dir, history)

    # Re-save the selected EMA weights as a compact inference checkpoint with
    # the fitted calibration/preprocessing contract.
    _save_inference_checkpoint(
        best_path,
        model,
        metadata,
        best_epoch,
        {"validation_macro_f1": validation_metrics["macro"]["f1"]},
    )
    source_checkpoint_sha256 = hashlib.sha256(best_path.read_bytes()).hexdigest()
    metadata["source_checkpoint_sha256"] = source_checkpoint_sha256
    metrics["checkpoint_size_bytes"] = best_path.stat().st_size
    metrics["source_checkpoint_sha256"] = source_checkpoint_sha256
    per_class_path = run_dir / "per_class_metrics.csv"
    with per_class_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["class", "precision", "recall", "f1", "support", "roc_auc"]
        )
        writer.writeheader()
        for class_name, values in metrics["per_class"].items():
            writer.writerow({"class": class_name, **values})
    report_lines = [
        f"# {architecture} Final Evaluation",
        "",
        f"- Best epoch: {best_epoch}",
        f"- Test loss: {metrics['test_loss']:.10f}",
        f"- Accuracy: {metrics['accuracy']:.10f}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.10f}",
        f"- Macro precision: {metrics['macro']['precision']:.10f}",
        f"- Macro recall: {metrics['macro']['recall']:.10f}",
        f"- Macro F1: {metrics['macro']['f1']:.10f}",
        f"- Macro ROC-AUC: {metrics.get('roc_auc_ovr_macro')}",
        f"- Test ECE before calibration: {calibration_report['test']['before']['expected_calibration_error']:.10f}",
        f"- Test ECE after calibration: {calibration_report['test']['after']['expected_calibration_error']:.10f}",
        f"- Temperature: {temperature:.10f}",
        f"- Misclassified images: {len(misclassified)}",
    ]
    (run_dir / "evaluation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    _atomic_json(run_dir / "metrics.json", metrics)
    parity_samples = []
    held_out_images, _held_out_labels = next(iter(loaders["val"]))
    parity_samples.append(("validation_held_out", held_out_images[:2].detach().cpu()))
    parity = export_and_verify_onnx(
        model,
        run_dir / "model.onnx",
        metadata,
        metadata["image_size"],
        config.output.onnx_opset,
        config.output.parity_atol,
        config.output.parity_rtol,
        warmups,
        iterations,
        parity_samples=parity_samples,
    )
    metrics["onnx_parity"] = parity
    metrics["onnx_size_bytes"] = (run_dir / "model.onnx").stat().st_size
    metrics["onnx_sha256"] = hashlib.sha256((run_dir / "model.onnx").read_bytes()).hexdigest()
    (run_dir / "checksum.sha256").write_text(
        f"{metrics['onnx_sha256']}  model.onnx\n", encoding="ascii"
    )
    _atomic_json(run_dir / "metrics.json", metrics)
    _atomic_json(
        run_dir / "run_state.json",
        {
            "status": "complete",
            "architecture": architecture,
            "best_epoch": best_epoch,
            "last_completed_epoch": int(history[-1]["epoch"]),
            "stopping_reason": (
                "early_stopping"
                if int(history[-1]["epoch"]) < config.optimization.epochs
                else "epoch_limit"
            ),
            "early_stopping_patience": config.runtime.early_stopping_patience,
            "completed_at": _utc_now(),
        },
    )
    return run_dir


def _train_architecture(
    config: TrainConfig,
    architecture: str,
    run_dir: Path,
    force_split: bool,
    resume: bool,
    resume_from: str | Path | None,
) -> Path:
    _apply_architecture_override(config, architecture)
    validate_config(config)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(config.runtime.seed)
    if config.runtime.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=config.runtime.deterministic_warn_only)
    device = config.resolved_device()

    split_manifest = build_split_manifest(config, force_split)
    class_to_idx = split_manifest["class_to_idx"]
    idx_to_class = split_manifest["idx_to_class"]
    num_classes = len(class_to_idx)
    model = build_model(
        architecture,
        num_classes,
        config.model.pretrained,
        config.model.dropout,
        config.model.drop_path_rate,
    ).to(device)
    preprocessing = resolve_preprocessing(
        model, config.data.image_size, config.data.use_model_preprocessing
    )
    config.data.image_size = int(preprocessing["image_size"])
    loaders, _ = create_dataloaders(
        config, force_split=False, manifest=split_manifest, preprocessing=preprocessing
    )
    split_path = Path(config.data.split_manifest)
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    signature = _training_signature(config, architecture, split_hash, preprocessing)
    metadata = {
        "schema_version": "2.5",
        "architecture": architecture,
        "num_classes": num_classes,
        "image_size": preprocessing["image_size"],
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "preprocessing": preprocessing,
        "normalization": {"mean": preprocessing["mean"], "std": preprocessing["std"]},
        "config": asdict(config),
        "created_at": _utc_now(),
        "split_manifest": split_path.as_posix(),
        "split_manifest_sha256": split_hash,
        "training_signature": signature,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "timm": timm.__version__,
            "albumentations": albumentations.__version__,
            "scikit_learn": sklearn.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
    }

    optimizer = create_optimizer_v2(
        model,
        opt="adamw",
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
        betas=tuple(config.optimization.betas),
        eps=config.optimization.eps,
        filter_bias_and_bn=config.optimization.filter_bias_and_norm_from_weight_decay,
        layer_decay=config.optimization.layer_decay,
    )
    optimizer_steps = math.ceil(
        len(loaders["train"]) / config.optimization.gradient_accumulation_steps
    )
    scheduler = _scheduler(
        optimizer,
        optimizer_steps,
        config.optimization.epochs,
        config.optimization.warmup_epochs,
        config.optimization.warmup_start_factor,
        config.optimization.min_learning_rate,
    )
    use_amp = config.runtime.mixed_precision and device.startswith("cuda")
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and config.runtime.amp_dtype == "float16",
    )
    ema = ModelEMA(model, config.runtime.ema_decay) if config.runtime.ema_enabled else None
    criterion = (
        build_class_weighted_loss(
            loaders["train"].dataset,
            num_classes,
            device,
            config.optimization.label_smoothing,
            config.optimization.class_weight_strategy,
            config.optimization.effective_num_beta,
        )
        if config.optimization.class_weighting
        else torch.nn.CrossEntropyLoss(label_smoothing=config.optimization.label_smoothing)
    )

    best_default = -float("inf") if config.runtime.early_stopping_mode == "max" else float("inf")
    history: list[dict] = []
    best_score = best_default
    best_epoch = 0
    bad_epochs = 0
    training_seconds = 0.0
    prior_peak_gpu_memory = None
    start_epoch = 1
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    checkpoint_path = Path(resume_from) if resume_from else last_path
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        _validate_resume_checkpoint(state, metadata)
        model.load_state_dict(state["training_model_state_dict"])
        if ema is not None and state.get("ema_state_dict") is not None:
            ema.load_state_dict(state["ema_state_dict"], state.get("ema_num_updates", 0))
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if state.get("scaler_state_dict"):
            scaler.load_state_dict(state["scaler_state_dict"])
        history = state.get("history", [])
        historical_fields = (
            "val_macro_precision",
            "val_macro_recall",
            "epoch_duration_seconds",
            "epoch_peak_gpu_memory_bytes",
            "maximum_gradient_norm",
            "optimizer_steps",
            "skipped_optimizer_steps",
        )
        for historical_row in history:
            for field in historical_fields:
                historical_row.setdefault(field, None)
        best_score = float(state.get("best_score", best_score))
        best_epoch = int(state.get("best_epoch", 0))
        bad_epochs = int(state.get("bad_epochs", 0))
        training_seconds = float(state.get("training_seconds", 0.0))
        prior_peak_gpu_memory = state.get("peak_gpu_memory_bytes")
        start_epoch = int(state["epoch"]) + 1
        _restore_rng_state(state.get("rng_state"))
        if state.get("train_generator_state") is not None:
            loaders["train"].crop_generator.set_state(
                _cpu_rng_tensor(state["train_generator_state"])
            )

    _atomic_json(
        run_dir / "run_state.json",
        {
            "status": "running",
            "architecture": architecture,
            "pid": os.getpid(),
            "started_or_resumed_at": _utc_now(),
            "start_epoch": start_epoch,
            "target_epochs": config.optimization.epochs,
        },
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    for epoch in range(start_epoch, config.optimization.epochs + 1):
        epoch_start = time.perf_counter()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        train_result = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
            scaler,
            scheduler,
            ema,
            use_amp,
            config.optimization.gradient_clip_norm,
            config.optimization.gradient_accumulation_steps,
            config.augmentation,
            config.runtime.amp_dtype,
        )
        evaluation_model = ema.module if ema is not None else model
        validation_result = run_epoch(evaluation_model, loaders["val"], criterion, device)
        monitor_value = _monitor_value(config, validation_result)
        validation_macro_f1 = float(
            f1_score(
                validation_result["targets"],
                validation_result["predictions"],
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        )
        validation_macro_precision = float(
            precision_score(
                validation_result["targets"],
                validation_result["predictions"],
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        )
        validation_macro_recall = float(
            recall_score(
                validation_result["targets"],
                validation_result["predictions"],
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        )
        epoch_duration = time.perf_counter() - epoch_start
        epoch_peak_gpu_memory = (
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else None
        )
        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "val_loss": validation_result["loss"],
            "val_accuracy": validation_result["accuracy"],
            "val_macro_f1": validation_macro_f1,
            "val_macro_precision": validation_macro_precision,
            "val_macro_recall": validation_macro_recall,
            "monitor_value": monitor_value,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "mixup_batches": train_result["batch_augmentation_counts"].get("mixup", 0),
            "cutmix_batches": train_result["batch_augmentation_counts"].get("cutmix", 0),
            "epoch_duration_seconds": epoch_duration,
            "epoch_peak_gpu_memory_bytes": epoch_peak_gpu_memory,
            "maximum_gradient_norm": train_result["maximum_gradient_norm"],
            "optimizer_steps": train_result["optimizer_steps"],
            "skipped_optimizer_steps": train_result["skipped_optimizer_steps"],
        }
        history.append(row)
        improved = _is_improved(config, monitor_value, best_score)
        if improved:
            best_score = monitor_value
            best_epoch = epoch
            bad_epochs = 0
            _save_inference_checkpoint(best_path, evaluation_model, metadata, epoch, row)
        else:
            bad_epochs += 1
        elapsed = training_seconds + (time.perf_counter() - wall_start)
        current_peak = epoch_peak_gpu_memory
        peak_gpu_memory = max(
            [value for value in (prior_peak_gpu_memory, current_peak) if value is not None],
            default=None,
        )
        _save_resume_checkpoint(
            last_path,
            model,
            ema,
            optimizer,
            scheduler,
            scaler,
            epoch,
            metadata,
            row,
            history,
            best_score,
            best_epoch,
            bad_epochs,
            elapsed,
            peak_gpu_memory,
            loaders["train"].crop_generator.get_state(),
        )
        _write_history(run_dir, history)
        print(
            f"[{architecture}] epoch {epoch}/{config.optimization.epochs} "
            f"train_loss={row['train_loss']:.5f} val_loss={row['val_loss']:.5f} "
            f"val_macro_f1={validation_macro_f1:.5f} best_epoch={best_epoch}",
            flush=True,
        )
        if bad_epochs >= config.runtime.early_stopping_patience:
            break

    training_seconds += time.perf_counter() - wall_start
    current_peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else None
    peak_gpu_memory = max(
        [value for value in (prior_peak_gpu_memory, current_peak) if value is not None],
        default=None,
    )
    if not best_path.exists():
        raise RuntimeError("No best checkpoint is available; train at least one epoch before finalization")
    optimizer = scheduler = scaler = ema = None
    if "evaluation_model" in locals():
        evaluation_model = None
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return _finalize_run(
        config,
        architecture,
        run_dir,
        model,
        loaders,
        criterion,
        metadata,
        history,
        best_path,
        best_epoch,
        training_seconds,
        peak_gpu_memory,
        device,
    )


def train_architecture(
    config: TrainConfig,
    architecture: str,
    force_split: bool = False,
    resume: bool = True,
    resume_from: str | Path | None = None,
) -> Path:
    if force_split and resume:
        raise ValueError("A split cannot be force-rebuilt while resume is enabled; use a new experiment and --no-resume")
    if resume_from is not None and not resume:
        raise ValueError("resume_from cannot be combined with resume=False")
    config = copy.deepcopy(config)
    run_dir = Path(config.output.run_root) / config.experiment_name / architecture
    with _run_lock(run_dir):
        try:
            return _train_architecture(
                config, architecture, run_dir, force_split, resume, resume_from
            )
        except Exception as exc:
            _atomic_json(
                run_dir / "run_state.json",
                {
                    "status": "failed",
                    "architecture": architecture,
                    "failed_at": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "resumable": (run_dir / "last.pt").exists(),
                },
            )
            raise


def train(
    config_path: str = "configs/training/phase2_5.yaml",
    architecture: str | None = None,
    force_split: bool = False,
    resume: bool = True,
    resume_from: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    return train_architecture(
        config,
        architecture or config.model.architecture,
        force_split,
        resume,
        resume_from,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a production crop-disease classifier.")
    parser.add_argument("--config", default="configs/training/phase2_5.yaml")
    parser.add_argument("--architecture", default=None)
    parser.add_argument(
        "--force-split",
        action="store_true",
        help="Explicitly rebuild the split manifest. Do not use for comparable benchmark runs.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Start over instead of resuming last.pt.")
    parser.add_argument("--resume-from", default=None, help="Resume from an explicit compatible checkpoint.")
    args = parser.parse_args()
    print(
        f"Training artifacts: {train(args.config, args.architecture, args.force_split, not args.no_resume, args.resume_from)}"
    )


if __name__ == "__main__":
    main()
