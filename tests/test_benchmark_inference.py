from __future__ import annotations

from scripts.benchmark_inference import (
    STAGE_KEYS,
    render_markdown,
    summarize_runs,
    validate_report_schema,
)


def _run(kind: str, index: int, base: float = 1.0) -> dict:
    return {
        "index": index,
        "kind": kind,
        "timings_ms": {
            stage: base + position
            for position, stage in enumerate(STAGE_KEYS)
        },
        "prediction": {
            "class_name": "Tomato_healthy",
            "confidence": 0.9,
            "top_3_predictions": [
                {"class_name": "Tomato_healthy", "confidence": 0.9}
            ],
            "mode": "onnx",
            "mock": False,
            "model_name": "EfficientNetV2-S",
            "model_version": "v1",
            "input_size": {"width": 300, "height": 300, "channels": 3},
        },
    }


def test_benchmark_report_schema_and_markdown_are_stable():
    cold = _run("cold", 0, 1.0)
    warmups = [_run("warmup", 1, 2.0)]
    measured = [_run("measured", 1, 3.0), _run("measured", 2, 4.0)]
    report = {
        "schema_version": "1.0",
        "generated_at": "2026-07-17T00:00:00+00:00",
        "command": ["scripts/benchmark_inference.py"],
        "config": {"warmup_runs": 1, "measured_runs": 2},
        "image": {
            "path": "data/example.jpg",
            "content_type": "image/jpeg",
            "format": "JPEG",
            "size_bytes": 128,
            "sha256": "abc",
        },
        "model": {
            "path": "models/releases/efficientnetv2_s_v1/model.onnx",
            "size_bytes": 80679586,
            "checksum_sha256": "abc",
            "architecture": "efficientnetv2_s",
            "model_name": "EfficientNetV2-S",
            "model_version": "v1",
            "input_size": {"width": 300, "height": 300, "channels": 3},
            "class_count": 15,
            "provider": "CPUExecutionProvider",
            "model_load_ms": 10.0,
        },
        "runtime": {
            "python": "3.11.0",
            "python_implementation": "CPython",
            "platform": "test",
            "machine": "AMD64",
            "processor": "test-cpu",
            "cpu_count": 8,
            "numpy": "2.0.0",
            "pillow": "12.0.0",
            "opencv": "5.0.0",
            "onnxruntime": "1.26.0",
            "onnxruntime_device": "CPU",
            "available_execution_providers": ["CPUExecutionProvider"],
            "session_execution_providers": ["CPUExecutionProvider"],
            "selected_execution_provider": "CPUExecutionProvider",
        },
        "memory": {
            "before": {"source": "test", "rss_bytes": 1, "peak_working_set_bytes": 2},
            "after": {"source": "test", "rss_bytes": 3, "peak_working_set_bytes": 4},
            "tracemalloc_current_bytes": 5,
            "tracemalloc_peak_bytes": 6,
        },
        "cold_run": cold,
        "warmup_runs": {
            "count": len(warmups),
            "summary_ms": summarize_runs(warmups),
            "runs": warmups,
        },
        "measured_runs": {
            "count": len(measured),
            "summary_ms": summarize_runs(measured),
            "runs": measured,
        },
    }

    validate_report_schema(report)
    markdown = render_markdown(report)

    assert "Leaflight Inference Baseline" in markdown
    assert "total_prediction" in markdown
    assert set(report["measured_runs"]["summary_ms"]) == set(STAGE_KEYS)
