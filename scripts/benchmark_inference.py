"""Benchmark the current Leaflight ONNX prediction pipeline.

The script measures the existing production-serving components without changing
model behavior: release-gated ``ModelService`` loading, metadata-driven
preprocessing, the active ONNX Runtime session, and the same response
postprocessing contract used by the API.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import mimetypes
import os
import platform
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.api.model_loader import ModelService
from backend.api.routes.predict import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_IMAGE_FORMATS,
    _validate_upload,
)
from backend.config import settings
from src.inference.preprocess_input import preprocess_for_onnx


SCHEMA_VERSION = "1.0"
STAGE_KEYS = (
    "image_decode_ms",
    "exif_correction_ms",
    "image_validation_ms",
    "preprocessing_ms",
    "onnx_inference_ms",
    "postprocessing_ms",
    "total_prediction_ms",
)
CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _timer_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def infer_content_type(path: Path) -> str | None:
    suffix_type = CONTENT_TYPES.get(path.suffix.lower())
    if suffix_type:
        return suffix_type
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed


def _validate_decode_and_correct(content: bytes, content_type: str | None) -> tuple[Image.Image, str | None, dict[str, float]]:
    """Mirror the API upload checks while timing decode and EXIF correction.

    The production route keeps these operations inside ``_validate_upload``.
    This script splits the same validation concerns into timed phases so the
    baseline can identify which part of the local prediction path is expensive.
    """

    timings: dict[str, float] = {}
    validation_start = time.perf_counter()
    if len(content) > settings.max_upload_size_bytes:
        raise ValueError(f"Image exceeds {settings.max_upload_size_mb}MB limit.")
    if not content:
        raise ValueError("Uploaded image is empty.")
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError("Supported image formats are JPEG, PNG, and WebP.")
    try:
        probe = Image.open(BytesIO(content))
        image_format = probe.format
        probe.verify()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a readable image.") from exc
    if image_format not in ALLOWED_IMAGE_FORMATS:
        raise ValueError("Supported image formats are JPEG, PNG, and WebP.")
    validation_ms = _timer_ms(validation_start)

    decode_start = time.perf_counter()
    try:
        decoded = Image.open(BytesIO(content))
        decoded.load()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a readable image.") from exc
    timings["image_decode_ms"] = _timer_ms(decode_start)

    exif_start = time.perf_counter()
    image = ImageOps.exif_transpose(decoded).convert("RGB")
    timings["exif_correction_ms"] = _timer_ms(exif_start)

    validation_start = time.perf_counter()
    if min(image.size) < 64:
        raise ValueError("Image is too small. Both dimensions must be at least 64 pixels.")
    if image.width * image.height > int(Image.MAX_IMAGE_PIXELS or 40_000_000):
        raise ValueError("Image dimensions exceed the 40-megapixel safety limit.")
    validation_ms += _timer_ms(validation_start)
    timings["image_validation_ms"] = validation_ms
    return image, image_format, timings


def _postprocess_prediction(service: ModelService, logits: np.ndarray) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (1, len(service.idx_to_class)) or not np.all(np.isfinite(logits)):
        raise RuntimeError(f"Unexpected ONNX output shape or values: {logits.shape}")
    probabilities = service._softmax(logits / float(service.temperature))[0]
    top_indices = probabilities.argsort()[-3:][::-1]
    top = [
        {
            "class_name": service.idx_to_class[int(index)],
            "confidence": float(probabilities[index]),
        }
        for index in top_indices
    ]
    return {
        "class_name": top[0]["class_name"],
        "confidence": top[0]["confidence"],
        "top_3_predictions": top,
        "mode": "onnx",
        "mock": False,
        "model_name": service.model_name,
        "model_version": service.model_version,
        "input_size": service.input_size,
    }


def run_prediction_once(
    *,
    service: ModelService,
    content: bytes,
    content_type: str | None,
    kind: str,
    index: int,
) -> dict[str, Any]:
    if (
        not service.loaded
        or service.session is None
        or service.input_name is None
        or service.output_name is None
        or service.image_size is None
    ):
        raise RuntimeError(service.load_error or "ONNX model is not loaded")

    total_start = time.perf_counter()
    image, _image_format, timings = _validate_decode_and_correct(content, content_type)

    start = time.perf_counter()
    model_input = preprocess_for_onnx(image, int(service.image_size), service.preprocessing)
    timings["preprocessing_ms"] = _timer_ms(start)

    start = time.perf_counter()
    logits = service.session.run([service.output_name], {service.input_name: model_input})[0]
    timings["onnx_inference_ms"] = _timer_ms(start)

    start = time.perf_counter()
    prediction = _postprocess_prediction(service, logits)
    timings["postprocessing_ms"] = _timer_ms(start)
    timings["total_prediction_ms"] = _timer_ms(total_start)

    return {
        "index": index,
        "kind": kind,
        "timings_ms": {key: float(timings[key]) for key in STAGE_KEYS},
        "prediction": prediction,
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for key in STAGE_KEYS:
        values = [float(run["timings_ms"][key]) for run in runs]
        if not values:
            summary[key] = {
                "count": 0,
                "p50": None,
                "p90": None,
                "p95": None,
                "mean": None,
                "min": None,
                "max": None,
            }
            continue
        summary[key] = {
            "count": len(values),
            "p50": float(statistics.median(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "mean": float(statistics.mean(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return summary


def _memory_snapshot() -> dict[str, int | str | None]:
    if os.name == "nt":
        try:
            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("page_fault_count", wintypes.DWORD),
                    ("peak_working_set_size", ctypes.c_size_t),
                    ("working_set_size", ctypes.c_size_t),
                    ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                    ("quota_paged_pool_usage", ctypes.c_size_t),
                    ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                    ("quota_non_paged_pool_usage", ctypes.c_size_t),
                    ("pagefile_usage", ctypes.c_size_t),
                    ("peak_pagefile_usage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            handle = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(
                handle,
                ctypes.byref(counters),
                counters.cb,
            )
            if ok:
                return {
                    "source": "windows_process_memory_info",
                    "rss_bytes": int(counters.working_set_size),
                    "peak_working_set_bytes": int(counters.peak_working_set_size),
                }
        except Exception:
            pass

    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system() != "Darwin":
            peak *= 1024
        return {
            "source": "resource_ru_maxrss",
            "rss_bytes": None,
            "peak_working_set_bytes": peak,
        }
    except Exception:
        return {
            "source": "not_available",
            "rss_bytes": None,
            "peak_working_set_bytes": None,
        }


def runtime_report(service: ModelService) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "opencv": cv2.__version__,
        "onnxruntime": ort.__version__,
        "onnxruntime_device": ort.get_device(),
        "available_execution_providers": ort.get_available_providers(),
        "session_execution_providers": service.session.get_providers() if service.session else [],
        "selected_execution_provider": service.provider,
    }


def build_report(
    *,
    image_path: Path,
    content: bytes,
    content_type: str | None,
    image_format: str | None,
    service: ModelService,
    model_load_ms: float,
    cold_run: dict[str, Any],
    warmup_runs: list[dict[str, Any]],
    measured_runs: list[dict[str, Any]],
    tracemalloc_current_bytes: int,
    tracemalloc_peak_bytes: int,
    memory_before: dict[str, Any],
    memory_after: dict[str, Any],
    created_at: datetime,
    argv: list[str],
) -> dict[str, Any]:
    model_path = service.model_path
    if model_path is None:
        raise RuntimeError("Model path is unavailable after loading")
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": created_at.isoformat(),
        "command": argv,
        "config": {
            "warmup_runs": len(warmup_runs),
            "measured_runs": len(measured_runs),
        },
        "image": {
            "path": _relative_or_absolute(image_path),
            "content_type": content_type,
            "format": image_format,
            "size_bytes": len(content),
            "sha256": _sha256_bytes(content),
        },
        "model": {
            "path": _relative_or_absolute(model_path),
            "size_bytes": model_path.stat().st_size,
            "checksum_sha256": service.model_checksum,
            "architecture": service.architecture,
            "model_name": service.model_name,
            "model_version": service.model_version,
            "input_size": service.input_size,
            "class_count": len(service.idx_to_class),
            "provider": service.provider,
            "model_load_ms": float(model_load_ms),
        },
        "runtime": runtime_report(service),
        "memory": {
            "before": memory_before,
            "after": memory_after,
            "tracemalloc_current_bytes": int(tracemalloc_current_bytes),
            "tracemalloc_peak_bytes": int(tracemalloc_peak_bytes),
        },
        "cold_run": cold_run,
        "warmup_runs": {
            "count": len(warmup_runs),
            "summary_ms": summarize_runs(warmup_runs),
            "runs": warmup_runs,
        },
        "measured_runs": {
            "count": len(measured_runs),
            "summary_ms": summarize_runs(measured_runs),
            "runs": measured_runs,
        },
    }
    validate_report_schema(report)
    return report


def validate_report_schema(report: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "generated_at",
        "command",
        "config",
        "image",
        "model",
        "runtime",
        "memory",
        "cold_run",
        "warmup_runs",
        "measured_runs",
    }
    missing = required - set(report)
    if missing:
        raise ValueError(f"Benchmark report is missing top-level keys: {sorted(missing)}")
    if report["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {report['schema_version']}")
    for key in STAGE_KEYS:
        if key not in report["cold_run"]["timings_ms"]:
            raise ValueError(f"Cold run is missing timing stage: {key}")
        if key not in report["warmup_runs"]["summary_ms"]:
            raise ValueError(f"Warmup summary is missing timing stage: {key}")
        if key not in report["measured_runs"]["summary_ms"]:
            raise ValueError(f"Measured summary is missing timing stage: {key}")
    if report["measured_runs"]["count"] < 1:
        raise ValueError("At least one measured run is required")


def render_markdown(report: dict[str, Any]) -> str:
    measured = report["measured_runs"]["summary_ms"]
    cold = report["cold_run"]["timings_ms"]
    lines = [
        "# Leaflight Inference Baseline",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Image: `{report['image']['path']}` ({report['image']['size_bytes']} bytes, {report['image']['content_type']})",
        (
            f"- Model: `{report['model']['model_name']} {report['model']['model_version']}` "
            f"via `{report['model']['provider']}`"
        ),
        f"- Model size: `{report['model']['size_bytes']}` bytes",
        f"- Warmup runs: `{report['warmup_runs']['count']}`",
        f"- Measured runs: `{report['measured_runs']['count']}`",
        "",
        "## Timing Summary",
        "",
        "| Stage | Cold ms | p50 ms | p90 ms | p95 ms | Mean ms | Min ms | Max ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in STAGE_KEYS:
        summary = measured[key]
        lines.append(
            "| {stage} | {cold:.3f} | {p50:.3f} | {p90:.3f} | {p95:.3f} | "
            "{mean:.3f} | {min:.3f} | {max:.3f} |".format(
                stage=key.removesuffix("_ms"),
                cold=float(cold[key]),
                p50=float(summary["p50"]),
                p90=float(summary["p90"]),
                p95=float(summary["p95"]),
                mean=float(summary["mean"]),
                min=float(summary["min"]),
                max=float(summary["max"]),
            )
        )
    memory = report["memory"]
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Python: `{report['runtime']['python']}`",
            f"- ONNX Runtime: `{report['runtime']['onnxruntime']}`",
            f"- NumPy: `{report['runtime']['numpy']}`",
            f"- Pillow: `{report['runtime']['pillow']}`",
            f"- OpenCV: `{report['runtime']['opencv']}`",
            f"- CPU count: `{report['runtime']['cpu_count']}`",
            f"- Processor: `{report['runtime']['processor']}`",
            f"- Available execution providers: `{report['runtime']['available_execution_providers']}`",
            f"- Session execution providers: `{report['runtime']['session_execution_providers']}`",
            "",
            "## Memory",
            "",
            f"- Process memory source: `{memory['after']['source']}`",
            f"- RSS after benchmark: `{memory['after']['rss_bytes']}` bytes",
            f"- Peak working set: `{memory['after']['peak_working_set_bytes']}` bytes",
            f"- Tracemalloc peak: `{memory['tracemalloc_peak_bytes']}` bytes",
            "",
            "## Prediction",
            "",
            f"- Class: `{report['cold_run']['prediction']['class_name']}`",
            f"- Confidence: `{report['cold_run']['prediction']['confidence']:.6f}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _unique_output_paths(output_dir: Path, created_at: datetime) -> tuple[Path, Path]:
    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"inference_baseline_{stamp}.json"
    markdown_path = output_dir / f"inference_baseline_{stamp}.md"
    suffix = 1
    while json_path.exists() or markdown_path.exists():
        json_path = output_dir / f"inference_baseline_{stamp}_{suffix}.json"
        markdown_path = output_dir / f"inference_baseline_{stamp}_{suffix}.md"
        suffix += 1
    return json_path, markdown_path


def write_outputs(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.fromisoformat(report["generated_at"])
    json_path, markdown_path = _unique_output_paths(output_dir, created_at)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Benchmark image does not exist: {image_path}")
    content = image_path.read_bytes()
    content_type = args.content_type or infer_content_type(image_path)

    # Production validation is invoked once before timing so benchmark input
    # acceptance is verified against the actual API helper.
    _validate_upload(content, content_type)
    image_format = Image.open(BytesIO(content)).format

    service = ModelService(args.model_path, args.metadata_path, args.manifest_path)
    memory_before = _memory_snapshot()
    tracemalloc.start()
    load_start = time.perf_counter()
    if not service.load():
        raise RuntimeError(service.load_error or "Failed to load ONNX model")
    model_load_ms = _timer_ms(load_start)

    cold_run = run_prediction_once(
        service=service,
        content=content,
        content_type=content_type,
        kind="cold",
        index=0,
    )
    warmup_runs = [
        run_prediction_once(
            service=service,
            content=content,
            content_type=content_type,
            kind="warmup",
            index=index + 1,
        )
        for index in range(args.warmups)
    ]
    measured_runs = [
        run_prediction_once(
            service=service,
            content=content,
            content_type=content_type,
            kind="measured",
            index=index + 1,
        )
        for index in range(args.runs)
    ]
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_after = _memory_snapshot()

    return build_report(
        image_path=image_path,
        content=content,
        content_type=content_type,
        image_format=image_format,
        service=service,
        model_load_ms=model_load_ms,
        cold_run=cold_run,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        tracemalloc_current_bytes=current_bytes,
        tracemalloc_peak_bytes=peak_bytes,
        memory_before=memory_before,
        memory_after=memory_after,
        created_at=_now_utc(),
        argv=sys.argv,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to a valid JPEG, PNG, or WebP image.")
    parser.add_argument("--content-type", default=None, help="Override inferred upload content type.")
    parser.add_argument("--warmups", type=int, default=5, help="Warmup predictions after the cold run.")
    parser.add_argument("--runs", type=int, default=20, help="Measured predictions after warmup.")
    parser.add_argument("--output-dir", default="artifacts/baselines", help="Directory for JSON and Markdown reports.")
    parser.add_argument("--model-path", default=None, help="Optional ONNX model path override.")
    parser.add_argument("--metadata-path", default=None, help="Optional model metadata path override.")
    parser.add_argument("--manifest-path", default=None, help="Optional release manifest path override.")
    args = parser.parse_args(argv)
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(args)
    json_path, markdown_path = write_outputs(report, Path(args.output_dir))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
