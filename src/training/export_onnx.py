from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def export_and_verify_onnx(
    model: torch.nn.Module, output_path: str | Path, metadata: dict[str, Any],
    image_size: int, opset: int = 18, atol: float = 1e-4, rtol: float = 1e-3,
    warmup_iterations: int = 10, latency_iterations: int = 50,
    parity_samples: list[tuple[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    cpu_model = model.eval().cpu()
    generator = torch.Generator(device="cpu").manual_seed(20250714)
    sample = torch.randn(2, 3, image_size, image_size, dtype=torch.float32, generator=generator)
    try:
        torch.onnx.export(
            cpu_model, sample, temporary, input_names=["images"], output_names=["logits"],
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset, do_constant_folding=True, dynamo=False,
        )
        onnx.checker.check_model(onnx.load(str(temporary)))
        session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
        temperature = float(metadata.get("calibration", {}).get("temperature", 1.0))
        max_abs_error = 0.0
        total_abs_error = 0.0
        total_values = 0
        max_probability_error = 0.0
        parity = True
        top1_match = True
        top3_match = True
        tested_inputs = [("synthetic_batch_1", sample[:1]), ("synthetic_batch_2", sample)]
        for name, extra_sample in parity_samples or []:
            if extra_sample.ndim != 4 or extra_sample.shape[1:] != sample.shape[1:]:
                raise ValueError(f"Parity sample {name!r} has incompatible shape {tuple(extra_sample.shape)}")
            tested_inputs.append((name, extra_sample.detach().cpu().float()))
        tested_batches = []
        for sample_name, test_sample in tested_inputs:
            with torch.inference_mode():
                torch_output = cpu_model(test_sample).numpy()
            onnx_output = session.run(["logits"], {"images": test_sample.numpy()})[0]
            absolute_error = np.abs(torch_output - onnx_output)
            max_abs_error = max(max_abs_error, float(np.max(absolute_error)))
            total_abs_error += float(absolute_error.sum())
            total_values += int(absolute_error.size)
            parity = parity and bool(np.allclose(torch_output, onnx_output, atol=atol, rtol=rtol))
            torch_probabilities = _softmax(torch_output / temperature)
            onnx_probabilities = _softmax(onnx_output / temperature)
            max_probability_error = max(
                max_probability_error,
                float(np.max(np.abs(torch_probabilities - onnx_probabilities))),
            )
            torch_top1 = np.argmax(torch_output, axis=1)
            onnx_top1 = np.argmax(onnx_output, axis=1)
            torch_top3 = np.argsort(torch_output, axis=1)[:, -3:][:, ::-1]
            onnx_top3 = np.argsort(onnx_output, axis=1)[:, -3:][:, ::-1]
            top1_match = top1_match and bool(np.array_equal(torch_top1, onnx_top1))
            top3_match = top3_match and bool(np.array_equal(torch_top3, onnx_top3))
            tested_batches.append({"name": sample_name, "batch_size": int(test_sample.shape[0])})
        parity = parity and top1_match and top3_match
        single = sample.numpy()[:1]
        for _ in range(int(warmup_iterations)):
            session.run(["logits"], {"images": single})
        timings = []
        for _ in range(int(latency_iterations)):
            start = time.perf_counter()
            session.run(["logits"], {"images": single})
            timings.append((time.perf_counter() - start) * 1000)
        median_ms = statistics.median(timings)
        parity_report = {
            "passed": parity,
            "max_absolute_error": max_abs_error,
            "mean_absolute_error": total_abs_error / max(total_values, 1),
            "max_calibrated_probability_error": max_probability_error,
            "atol": atol,
            "rtol": rtol,
            "tested_batch_sizes": sorted({row["batch_size"] for row in tested_batches}),
            "tested_inputs": tested_batches,
            "top1_class_match": top1_match,
            "top3_class_order_match": top3_match,
            "onnx_cpu_inference": {
                "median_latency_ms": median_ms,
                "mean_latency_ms": statistics.mean(timings),
                "p90_latency_ms": float(np.percentile(timings, 90)),
                "p95_latency_ms": float(np.percentile(timings, 95)),
                "images_per_second": 1000.0 / median_ms,
                "warmup_iterations": int(warmup_iterations),
                "iterations": int(latency_iterations),
            },
        }
        if not parity:
            raise RuntimeError(f"ONNX parity check failed: {parity_report}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    bundle_metadata = {**metadata, "onnx": {"opset": opset, "input_name": "images", "output_name": "logits", "parity": parity_report}}
    metadata_output = output.with_suffix(".json")
    metadata_temporary = metadata_output.with_suffix(".json.tmp")
    metadata_temporary.write_text(json.dumps(bundle_metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(metadata_temporary, metadata_output)
    return parity_report


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)
