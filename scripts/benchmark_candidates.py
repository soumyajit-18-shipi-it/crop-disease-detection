"""Benchmark completed Phase 2.5 candidates under one explicit methodology."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import onnxruntime as ort
import torch

from src.models.model_factory import build_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(timings: list[float], batch_size: int, warmups: int) -> dict:
    median = float(statistics.median(timings))
    return {
        "median_latency_ms": median,
        "mean_latency_ms": float(statistics.mean(timings)),
        "p90_latency_ms": float(np.percentile(timings, 90)),
        "p95_latency_ms": float(np.percentile(timings, 95)),
        "throughput_images_per_second": float(batch_size * 1000.0 / median),
        "batch_size": batch_size,
        "warmup_iterations": warmups,
        "iterations": len(timings),
    }


def _pytorch_benchmark(model, sample, device: str, warmups: int, iterations: int) -> dict:
    model = model.eval().to(device)
    sample = sample.to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(warmups):
            model(sample)
        if device == "cuda":
            torch.cuda.synchronize()
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(sample)
            if device == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    result = _summary(timings, sample.shape[0], warmups)
    result["device"] = device
    result["peak_cuda_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
    )
    return result


def _onnx_benchmark(path: Path, sample: np.ndarray, warmups: int, iterations: int, threads: int) -> dict:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for _ in range(warmups):
        session.run([output_name], {input_name: sample})
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run([output_name], {input_name: sample})
        timings.append((time.perf_counter() - start) * 1000.0)
    result = _summary(timings, sample.shape[0], warmups)
    result.update(
        {
            "provider": "CPUExecutionProvider",
            "intra_op_num_threads": threads,
            "inter_op_num_threads": 1,
            "execution_mode": "sequential",
        }
    )
    return result


def benchmark_candidate(
    run_root: Path,
    architecture: str,
    warmups: int,
    iterations: int,
    batch_size: int,
    cpu_threads: int,
) -> dict:
    run_dir = run_root / architecture
    metadata = json.loads((run_dir / "model.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model_config = metadata["config"]["model"]
    model = build_model(
        architecture,
        int(metadata["num_classes"]),
        False,
        float(model_config["dropout"]),
        float(model_config["drop_path_rate"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    image_size = int(metadata["image_size"])
    generator = torch.Generator(device="cpu").manual_seed(20260715)
    sample = torch.randn(
        batch_size, 3, image_size, image_size, generator=generator, dtype=torch.float32
    )

    cuda_result = _pytorch_benchmark(model, sample, "cuda", warmups, iterations)
    model.to("cpu")
    torch.cuda.empty_cache()
    cpu_result = _pytorch_benchmark(model, sample, "cpu", warmups, iterations)
    onnx_result = _onnx_benchmark(
        run_dir / "model.onnx", sample.numpy(), warmups, iterations, cpu_threads
    )
    del model, checkpoint, sample
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "architecture": architecture,
        "image_size": image_size,
        "parameter_count": parameter_count,
        "checkpoint_size_bytes": (run_dir / "best.pt").stat().st_size,
        "onnx_size_bytes": (run_dir / "model.onnx").stat().st_size,
        "onnx_sha256": _sha256(run_dir / "model.onnx"),
        "pytorch_cuda": cuda_result,
        "pytorch_cpu": cpu_result,
        "onnxruntime_cpu": onnx_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", default="artifacts/training/crop_disease_phase2_5"
    )
    parser.add_argument(
        "--architectures", nargs="+", default=["efficientnetv2_s", "convnext_tiny"]
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The fair benchmark requires the CUDA environment used for training")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    run_root = Path(args.run_root)
    output = Path(args.output) if args.output else run_root / "fair_benchmark.json"
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "hardware": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "onnxruntime": ort.__version__,
            "batch_size": args.batch_size,
            "warmup_iterations": args.warmups,
            "iterations": args.iterations,
            "pytorch_cpu_threads": args.cpu_threads,
            "onnx_provider": "CPUExecutionProvider",
            "onnx_intra_op_threads": args.cpu_threads,
            "onnx_inter_op_threads": 1,
            "onnx_execution_mode": "sequential",
            "input_policy": "each model's recorded production-native input size",
        },
        "candidates": {},
    }
    for architecture in args.architectures:
        print(f"Benchmarking {architecture}...", flush=True)
        report["candidates"][architecture] = benchmark_candidate(
            run_root,
            architecture,
            args.warmups,
            args.iterations,
            args.batch_size,
            args.cpu_threads,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
