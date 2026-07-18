from __future__ import annotations

from scripts.benchmark_prediction_api import (
    render_markdown,
    summarize_values,
    validate_report_schema,
)


def _scenario(name: str, concurrency: int) -> dict:
    return {
        "name": name,
        "concurrency": concurrency,
        "requests": 2,
        "success_count": 2,
        "error_count": 0,
        "status_counts": {"200": 2},
        "wall_time_ms": 100.0,
        "throughput_requests_per_second": 20.0,
        "process_cpu_seconds": 0.02,
        "process_cpu_ms_per_request": 10.0,
        "latency_summary_ms": summarize_values([40.0, 50.0]),
        "success_latency_summary_ms": summarize_values([40.0, 50.0]),
        "model_only_p50_ms": 30.0,
        "api_over_model_p50_delta_ms": 15.0,
        "database_write_summary_ms": summarize_values([2.0, 3.0]),
        "database_write_timing_count": 2,
        "database_rows_before": 0,
        "database_rows_after": 2,
        "database_rows_inserted": 2,
        "memory_before": {"source": "test", "rss_bytes": 1, "peak_working_set_bytes": 1},
        "memory_after": {"source": "test", "rss_bytes": 2, "peak_working_set_bytes": 2},
        "errors": [],
    }


def test_api_prediction_benchmark_report_schema_and_markdown_are_stable():
    report = {
        "schema_version": "1.0",
        "generated_at": "2026-07-17T00:00:00+00:00",
        "command": ["scripts/benchmark_prediction_api.py"],
        "config": {
            "requests_per_level": 2,
            "concurrency_levels": [1, 2],
            "model_warmups": 1,
            "model_runs": 2,
        },
        "image": {
            "path": "data/example.jpg",
            "content_type": "image/jpeg",
            "size_bytes": 128,
            "sha256": "abc",
        },
        "database": {
            "backend": "temporary_sqlite",
            "url_kind": "sqlite",
            "path": None,
            "session_user_id": "benchmark-user",
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
        },
        "runtime": {
            "python": "3.11.0",
            "python_implementation": "CPython",
            "platform": "test",
            "machine": "AMD64",
            "processor": "test-cpu",
            "cpu_count": 8,
            "onnxruntime": "1.26.0",
            "onnxruntime_device": "CPU",
            "available_execution_providers": ["CPUExecutionProvider"],
            "session_execution_providers": ["CPUExecutionProvider"],
            "selected_execution_provider": "CPUExecutionProvider",
        },
        "model_only": {
            "cold_ms": 35.0,
            "warmup_runs": 1,
            "warmup_summary_ms": summarize_values([31.0]),
            "measured_runs": 2,
            "measured_summary_ms": summarize_values([30.0, 32.0]),
            "prediction": {
                "class_name": "Tomato_healthy",
                "confidence": 0.9,
                "model_name": "EfficientNetV2-S",
                "model_version": "v1",
                "input_size": {"width": 300, "height": 300, "channels": 3},
            },
        },
        "api": {
            "sequential": _scenario("sequential", 1),
            "concurrency": [_scenario("concurrency_1", 1), _scenario("concurrency_2", 2)],
        },
    }

    validate_report_schema(report)
    markdown = render_markdown(report)

    assert "Leaflight API Prediction Baseline" in markdown
    assert "API minus model" in markdown
    assert "concurrency_2" in markdown
