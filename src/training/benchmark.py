from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

import torch

from src.training.config import SelectionConfig, load_config


def _minmax(values: dict[str, float], higher_is_better: bool = True) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high == low:
        return {key: 1.0 for key in values}
    normalized = {key: (value - low) / (high - low) for key, value in values.items()}
    return normalized if higher_is_better else {key: 1.0 - value for key, value in normalized.items()}


def _eligible_results(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    eligible = {
        name: metrics
        for name, metrics in results.items()
        if metrics.get("onnx_parity", {}).get("passed") is True
        and metrics.get("validation", {}).get("macro", {}).get("f1") is not None
        and metrics.get("validation", {}).get("calibration", {}).get("expected_calibration_error") is not None
        and metrics.get("onnx_parity", {}).get("onnx_cpu_inference", {}).get("images_per_second") is not None
        and metrics.get("onnx_size_bytes") is not None
    }
    memory_present = [metrics.get("peak_gpu_memory_bytes") is not None for metrics in eligible.values()]
    # A mixed CPU/GPU benchmark is not comparable. All-CPU runs retain the
    # memory component as an equal constant instead of inventing measurements.
    if any(memory_present) and not all(memory_present):
        return {}
    return eligible


def _score_candidates(
    results: dict[str, dict[str, Any]], selection: SelectionConfig | None = None
) -> tuple[dict[str, float], dict[str, dict[str, float]], bool]:
    selection = selection or SelectionConfig()
    eligible = _eligible_results(results)
    if not eligible:
        return {}, {}, False
    f1 = {name: float(value["validation"]["macro"]["f1"]) for name, value in eligible.items()}
    ece = {
        name: float(value["validation"]["calibration"]["expected_calibration_error"])
        for name, value in eligible.items()
    }
    speed = {
        name: float(value["onnx_parity"]["onnx_cpu_inference"]["images_per_second"])
        for name, value in eligible.items()
    }
    size = {name: float(value["onnx_size_bytes"]) for name, value in eligible.items()}
    has_memory = all(value.get("peak_gpu_memory_bytes") is not None for value in eligible.values())
    memory = (
        {name: float(value["peak_gpu_memory_bytes"]) for name, value in eligible.items()}
        if has_memory
        else {}
    )
    normalized = {
        "validation_macro_f1": _minmax(f1),
        "calibration_quality": _minmax(ece, False),
        "inference_speed": _minmax(speed),
        "model_size": _minmax(size, False),
        "memory_usage": _minmax(memory, False) if memory else {name: 1.0 for name in eligible},
    }
    weights = {
        "validation_macro_f1": selection.validation_macro_f1_weight,
        "calibration_quality": selection.calibration_weight,
        "inference_speed": selection.inference_speed_weight,
        "model_size": selection.model_size_weight,
        "memory_usage": selection.memory_weight,
    }
    components = {
        name: {criterion: normalized[criterion][name] for criterion in normalized}
        for name in eligible
    }
    scores = {
        name: sum(weights[criterion] * components[name][criterion] for criterion in weights)
        for name in eligible
    }
    return scores, components, has_memory


def _select_best(
    results: dict[str, dict[str, Any]], selection: SelectionConfig | None = None
) -> tuple[str | None, dict[str, float]]:
    scores, _components, _has_memory = _score_candidates(results, selection)
    if not scores:
        return None, {}
    return max(scores, key=scores.get), scores


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f"{destination.name}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _selection_weights(selection: SelectionConfig) -> dict[str, float]:
    return {
        "validation_macro_f1": selection.validation_macro_f1_weight,
        "calibration_quality": selection.calibration_weight,
        "inference_speed": selection.inference_speed_weight,
        "model_size": selection.model_size_weight,
        "memory_usage": selection.memory_weight,
    }


def _promote(
    config,
    architecture: str,
    metrics: dict[str, Any],
    scores: dict[str, float],
    components: dict[str, dict[str, float]],
    has_memory: bool,
) -> Path:
    run = Path(config.output.run_root) / config.experiment_name / architecture
    output = Path(config.output.run_root) / config.experiment_name / "production"
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "best.pt": "best.pt",
        "model.onnx": "best.onnx",
        "training_history.csv": "training_history.csv",
        "confusion_matrix.png": "confusion_matrix.png",
        "classification_report.json": "classification_report.json",
        "calibration.json": "calibration.json",
        "reliability_diagram.png": "reliability_diagram.png",
    }
    missing = [source for source in files if not (run / source).is_file()]
    if not (run / "model.json").is_file():
        missing.append("model.json")
    if missing:
        raise FileNotFoundError(f"Cannot promote incomplete {architecture} run; missing: {missing}")
    for source, destination in files.items():
        _atomic_copy(run / source, output / destination)

    selection_payload = {
        "selected_architecture": architecture,
        "score": scores[architecture],
        "weights": _selection_weights(config.selection),
        "normalized_components": components[architecture],
        "memory_measurement": "peak_gpu_training_memory" if has_memory else "unavailable_equal_component",
        "candidate_scores": scores,
    }
    metadata = json.loads((run / "model.json").read_text(encoding="utf-8"))
    metadata["production_selection"] = selection_payload
    _atomic_json(output / "metadata.json", metadata)
    promoted_metrics = {**metrics, "selection": selection_payload}
    _atomic_json(output / "metrics.json", promoted_metrics)
    manifest = {
        "schema_version": "1.0",
        "architecture": architecture,
        "files": {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "bundle_manifest.json"
        },
    }
    _atomic_json(output / "bundle_manifest.json", manifest)
    return output


def _run_status(run_dir: Path, metrics_path: Path) -> str:
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("onnx_parity", {}).get("passed") is not True:
                return "export/parity failed"
            required = {
                "best.pt", "model.onnx", "model.json", "training_history.csv",
                "confusion_matrix.png", "classification_report.json", "calibration.json",
            }
            if not all((run_dir / name).is_file() for name in required):
                return "incomplete bundle"
            return "complete"
        except (OSError, json.JSONDecodeError):
            return "invalid metrics"
    state_path = run_dir / "run_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") == "running":
                try:
                    pid = int(state.get("pid", -1))
                    if pid <= 0:
                        raise ValueError("invalid pid")
                    os.kill(pid, 0)
                    return "running"
                except (OSError, ValueError):
                    return "resumable" if (run_dir / "last.pt").exists() else "interrupted"
            if state.get("status") == "failed":
                return "failed (resumable)" if state.get("resumable") else "failed"
        except (OSError, json.JSONDecodeError):
            pass
    if (run_dir / "last.pt").exists():
        return "resumable"
    return "not started"


def _write_comparison_report(
    config,
    candidates: list[str],
    results: dict[str, dict[str, Any]],
    statuses: dict[str, str],
    selected: str | None,
    scores: dict[str, float],
    production_dir: Path | None,
    split_consistent: bool,
) -> Path:
    comparison = Path("docs/model_comparison.md")
    lines = [
        "# Production Model Comparison",
        "",
        "Every candidate uses the persisted split manifest. Missing values are shown as `-`; no metric is estimated or backfilled.",
        "",
        "| Architecture | Status | Input px | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Balanced Acc. | MCC | Kappa | ROC-AUC | Val ECE | ONNX img/s | ONNX MB | Peak GPU GB | Train min | Score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in candidates:
        metrics = results.get(name, {})
        peak_memory = metrics.get("peak_gpu_memory_bytes")
        row = [
            name,
            statuses[name],
            _fmt(metrics.get("image_size")),
            _fmt(metrics.get("accuracy")),
            _fmt(metrics.get("macro", {}).get("precision")),
            _fmt(metrics.get("macro", {}).get("recall")),
            _fmt(metrics.get("macro", {}).get("f1")),
            _fmt(metrics.get("weighted", {}).get("f1")),
            _fmt(metrics.get("balanced_accuracy")),
            _fmt(metrics.get("matthews_correlation_coefficient")),
            _fmt(metrics.get("cohen_kappa")),
            _fmt(metrics.get("roc_auc_ovr_macro")),
            _fmt(metrics.get("validation", {}).get("calibration", {}).get("expected_calibration_error")),
            _fmt(metrics.get("onnx_parity", {}).get("onnx_cpu_inference", {}).get("images_per_second"), 2),
            _fmt(metrics.get("onnx_size_bytes") / 1048576 if metrics.get("onnx_size_bytes") else None, 2),
            _fmt(peak_memory / 1073741824 if peak_memory is not None else None, 2),
            _fmt(metrics.get("training_time_seconds") / 60 if metrics.get("training_time_seconds") else None, 1),
            _fmt(scores.get(name)),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", "## Selection", ""])
    if selected:
        selected_metrics = results[selected]
        lines.extend(
            [
                f"Selected `{selected}` with a production score of `{scores[selected]:.4f}`.",
                "",
                "The score uses 40% validation macro F1, 20% inverse validation ECE after temperature scaling, 15% ONNX CPU throughput, 15% inverse ONNX size, and 10% inverse peak GPU memory. Metrics are min-max normalized only across eligible candidates.",
                "",
                f"Its measured validation macro F1 is `{selected_metrics['validation']['macro']['f1']:.4f}`, calibrated validation ECE is `{selected_metrics['validation']['calibration']['expected_calibration_error']:.4f}`, and ONNX CPU throughput is `{selected_metrics['onnx_parity']['onnx_cpu_inference']['images_per_second']:.2f}` images/s on the benchmark host.",
                "",
                f"Production bundle: `{production_dir.as_posix()}`.",
            ]
        )
    else:
        reasons = []
        if len(results) != len(candidates) and config.selection.require_all_candidates:
            reasons.append("all configured candidate runs have not completed")
        if not split_consistent:
            reasons.append("completed runs do not share one split-manifest hash")
        reason_text = "; ".join(reasons) or "no complete parity-verified candidate set is available"
        lines.append(f"No production model has been selected because {reason_text}.")
    comparison.parent.mkdir(parents=True, exist_ok=True)
    comparison.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison


def _write_training_report(config, candidates, results, statuses, selected) -> Path:
    output = Path("docs/training_results.md")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        hardware = f"- GPU: {properties.name}, {properties.total_memory / 1073741824:.1f} GB VRAM."
    else:
        hardware = "- GPU: unavailable to PyTorch; GPU-memory values will remain unreported."
    lines = [
        "# Phase 2.5 Training Results",
        "",
        "> This file reports only artifacts currently present on disk. A blank table cell is never interpreted as a benchmark result.",
        "",
        f"- Platform: `{platform.platform()}`",
        f"- PyTorch: `{torch.__version__}`; CUDA available: `{str(cuda_available).lower()}`",
        hardware,
        f"- Persisted split: `{config.data.split_manifest}`",
        f"- Production selection: `{selected}`" if selected else "- Production selection: pending",
        "",
        "## Pipeline Audit and Implemented Policy",
        "",
        "- AdamW excludes bias and normalization parameters from weight decay; per-step cosine decay includes linear warmup and a non-zero minimum LR.",
        "- Full pretrained fine-tuning uses each timm backbone's native resolution, interpolation, mean, std, and evaluation crop contract.",
        "- EMA, CUDA AMP, gradient accumulation, overflow-aware scheduler stepping, clipping, label smoothing, effective-number class weighting, and optional balanced sampling are configurable.",
        "- Field augmentations cover restrained geometry, illumination/color, CLAHE, blur/defocus, shadow/light fog, JPEG degradation, and partial occlusion. Artificial rain is supported but disabled pending field evidence.",
        "- MixUp and CutMix are batch-level and mutually sampled. Weighted loss is enabled; balanced sampling is disabled in the preset to avoid double-correcting class imbalance.",
        "- Deterministic RNG and DataLoader generator states are checkpointed. `last.pt` is an atomic resume checkpoint; `best.pt` is a compact EMA inference checkpoint selected by validation macro F1.",
        "- Temperature scaling is fit on validation logits. The test split is evaluated only after model selection within each run, and raw/calibrated ECE are both retained.",
        "",
        "## Candidate Status",
        "",
    ]
    for name in candidates:
        metrics = results.get(name)
        lines.extend([f"### {name}", "", f"- Status: {statuses[name]}"])
        if metrics:
            lines.extend(
                [
                    f"- Best epoch: {_fmt(metrics.get('best_epoch'))}",
                    f"- Training time: {_fmt(metrics.get('training_time_seconds'), 1)} seconds",
                    f"- Peak GPU memory: {_fmt(metrics.get('peak_gpu_memory_bytes'))} bytes",
                    f"- PyTorch CPU throughput: {_fmt(metrics.get('cpu_inference', {}).get('images_per_second'), 2)} images/s",
                    f"- ONNX CPU throughput: {_fmt(metrics.get('onnx_parity', {}).get('onnx_cpu_inference', {}).get('images_per_second'), 2)} images/s",
                    f"- Compact checkpoint: {_fmt(metrics.get('checkpoint_size_bytes'))} bytes",
                    f"- ONNX: {_fmt(metrics.get('onnx_size_bytes'))} bytes",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## Run or Resume Exactly",
            "",
            "All candidates resume `last.pt` by default and reuse the existing split:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe -m src.training.benchmark --config configs/training/phase2_5.yaml --train",
            "```",
            "",
            "Resume one candidate:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe -m src.training.train --config configs/training/phase2_5.yaml --architecture convnext_base",
            "```",
            "",
            "Do not pass `--force-split` for these runs. Resume validation rejects a changed split hash, architecture, preprocessing contract, or optimization signature.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _write_production_report(config, selected, results, scores, production_dir, has_memory: bool) -> Path:
    output = Path("docs/production_model.md")
    lines = ["# Production Model", ""]
    if selected:
        metrics = results[selected]
        memory_description = (
            "measured peak GPU memory (10%)"
            if has_memory
            else "a 10% memory component held equal because GPU memory was unavailable"
        )
        lines.extend(
            [
                f"The selected production classifier is `{selected}`.",
                "",
                f"It won the required multi-objective score (`{scores[selected]:.4f}`) after all candidates completed on one persisted split and passed PyTorch/ONNX parity. The decision balances validation macro F1 (40%), calibrated confidence quality (20%), ONNX CPU speed (15%), ONNX size (15%), and {memory_description}.",
                "",
                f"- Validation macro F1: `{metrics['validation']['macro']['f1']:.4f}`",
                f"- Validation ECE after temperature scaling: `{metrics['validation']['calibration']['expected_calibration_error']:.4f}`",
                f"- Test macro F1 (reported after training): `{metrics['macro']['f1']:.4f}`",
                f"- ONNX parity max absolute logit error: `{metrics['onnx_parity']['max_absolute_error']:.6g}`",
                f"- Bundle: `{production_dir.as_posix()}`",
                "",
                "The production metadata carries the exact resize, crop, interpolation, normalization, and temperature values used by the backend. Confidence is therefore calibrated consistently in PyTorch and ONNX serving.",
            ]
        )
        if not has_memory:
            lines.append("")
            lines.append("GPU memory was unavailable for every candidate, so the 10% memory component was held equal rather than populated with estimated values.")
    else:
        lines.extend(
            [
                "No production model is selected yet.",
                "",
                "Selection remains intentionally blocked until EfficientNetV2-S, ConvNeXt-Tiny, and ConvNeXt-Base all finish using the unchanged split and pass ONNX parity. This avoids presenting a partial benchmark winner as the production choice.",
                "",
                "Run or resume the benchmark with:",
                "",
                "```powershell",
                ".\\.venv\\Scripts\\python.exe -m src.training.benchmark --config configs/training/phase2_5.yaml --train",
                "```",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_reports(
    config_path: str,
    train_missing: bool = False,
    include_optional: bool = False,
) -> tuple[Path, Path, str | None]:
    config = load_config(config_path)
    candidates = list(config.model.architectures)
    if include_optional:
        candidates.extend(name for name in config.model.optional_architectures if name not in candidates)
    results: dict[str, dict[str, Any]] = {}
    statuses: dict[str, str] = {}
    for architecture in candidates:
        run_dir = Path(config.output.run_root) / config.experiment_name / architecture
        metrics_path = run_dir / "metrics.json"
        existing_metrics = None
        if metrics_path.exists():
            try:
                existing_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing_metrics = None
        needs_training = (
            not existing_metrics
            or existing_metrics.get("onnx_parity", {}).get("passed") is not True
            or _run_status(run_dir, metrics_path) != "complete"
        )
        if train_missing and needs_training:
            try:
                from src.training.train import train_architecture

                train_architecture(config, architecture, resume=True)
            except Exception as exc:
                print(f"{architecture} failed: {type(exc).__name__}: {exc}", flush=True)
        if metrics_path.exists():
            try:
                results[architecture] = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        statuses[architecture] = _run_status(run_dir, metrics_path)

    expected_split_hash = _sha256(Path(config.data.split_manifest))
    result_split_hashes = [metrics.get("split_manifest_sha256") for metrics in results.values()]
    split_consistent = all(value == expected_split_hash for value in result_split_hashes)
    enough_results = (
        all(statuses[name] == "complete" for name in candidates)
        if config.selection.require_all_candidates
        else any(status == "complete" for status in statuses.values())
    )
    selected = None
    scores: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    has_memory = False
    if enough_results and split_consistent:
        selection_results = {
            name: results[name]
            for name in candidates
            if statuses[name] == "complete" and name in results
        }
        scores, components, has_memory = _score_candidates(selection_results, config.selection)
        selected = max(scores, key=scores.get) if scores else None
    production_dir = (
        _promote(config, selected, results[selected], scores, components, has_memory)
        if selected
        else None
    )
    comparison = _write_comparison_report(
        config,
        candidates,
        results,
        statuses,
        selected,
        scores,
        production_dir,
        split_consistent,
    )
    training = _write_training_report(config, candidates, results, statuses, selected)
    _write_production_report(config, selected, results, scores, production_dir, has_memory)
    return comparison, training, selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, benchmark, and select the Phase 2.5 production model.")
    parser.add_argument("--config", default="configs/training/phase2_5.yaml")
    parser.add_argument("--train", action="store_true", help="Train or resume architectures with missing metrics.")
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also benchmark configured optional architectures such as Swin-Tiny.",
    )
    args = parser.parse_args()
    comparison, training, selected = write_reports(args.config, args.train, args.include_optional)
    print(comparison)
    print(training)
    print(f"selected={selected or 'none'}")


if __name__ == "__main__":
    main()
