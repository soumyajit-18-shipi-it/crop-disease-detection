"""Benchmark Leaflight's authenticated /predict API path.

The benchmark uses the real FastAPI app, a temporary SQLite database by
default, real session/CSRF creation, and the active ONNX model service. It does
not change production authentication code.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
from ctypes import wintypes
import hashlib
import json
import logging
import mimetypes
import os
import platform
import statistics
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


SCHEMA_VERSION = "1.0"
CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
DEFAULT_CONCURRENCY_LEVELS = (1, 2, 5, 10)


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


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile for an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "p50": None,
            "p90": None,
            "p95": None,
            "mean": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "p50": float(statistics.median(values)),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "mean": float(statistics.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


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
            ok = psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(),
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


def _configure_benchmark_environment(database_url: str) -> None:
    os.environ["DATABASE_URL"] = database_url
    os.environ["ENVIRONMENT"] = "test"
    os.environ["AUTH_SECRET"] = "benchmark-auth-secret-with-at-least-32-characters"
    os.environ["COOKIE_SECURE"] = "false"
    os.environ["COOKIE_SAMESITE"] = "lax"
    os.environ["LOG_TO_FILE"] = "false"
    os.environ.setdefault("APP_URL", "http://127.0.0.1:5173")
    os.environ.setdefault("OAUTH_CALLBACK_URL", "http://127.0.0.1:8000/auth/google/callback")
    os.environ.setdefault("GOOGLE_CLIENT_ID", "benchmark-client-id")
    os.environ.setdefault("GOOGLE_CLIENT_SECRET", "benchmark-client-secret")
    os.environ.setdefault("CORS_ORIGINS", "http://127.0.0.1:5173")


def _load_backend_modules() -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from backend.api.auth import CSRF_COOKIE, SESSION_COOKIE, create_session, isoformat, utc_now
    from backend.api.model_loader import model_service
    from backend.api.routes import predict as predict_route
    from backend.api.routes.predict import _validate_upload
    from backend.db.database import connect_database
    from backend.main import app

    return {
        "TestClient": TestClient,
        "CSRF_COOKIE": CSRF_COOKIE,
        "SESSION_COOKIE": SESSION_COOKIE,
        "create_session": create_session,
        "isoformat": isoformat,
        "utc_now": utc_now,
        "model_service": model_service,
        "predict_route": predict_route,
        "_validate_upload": _validate_upload,
        "connect_database": connect_database,
        "app": app,
    }


def _create_authenticated_benchmark_session(client: Any, modules: dict[str, Any]) -> dict[str, str]:
    user_id = "benchmark-user"
    now = modules["isoformat"](modules["utc_now"]())
    with modules["connect_database"]() as connection:
        connection.execute(
            """
            INSERT INTO users(
                id, name, email, auth_provider, provider_account_id,
                created_at, last_login_at
            ) VALUES (?, ?, ?, 'google', ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                email = excluded.email,
                last_login_at = excluded.last_login_at
            """,
            (
                user_id,
                "Benchmark Farmer",
                "benchmark@example.test",
                "google-benchmark-user",
                now,
                now,
            ),
        )
        connection.commit()
    session_token, csrf_token, expires_at = modules["create_session"](user_id)
    client.cookies.set(modules["SESSION_COOKIE"], session_token)
    client.cookies.set(modules["CSRF_COOKIE"], csrf_token)
    client.headers["X-CSRF-Token"] = csrf_token
    return {
        "user_id": user_id,
        "session_expires_at": expires_at,
    }


def _scan_count(modules: dict[str, Any], user_id: str) -> int:
    with modules["connect_database"]() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM scans WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return int(row["count"])


def _run_model_only(
    *,
    model_service: Any,
    image: Any,
    warmups: int,
    runs: int,
) -> dict[str, Any]:
    cold_start = time.perf_counter()
    cold_prediction = model_service.predict(image)
    cold_ms = _timer_ms(cold_start)

    warmup_timings = []
    for _ in range(warmups):
        start = time.perf_counter()
        model_service.predict(image)
        warmup_timings.append(_timer_ms(start))

    measured_timings = []
    for _ in range(runs):
        start = time.perf_counter()
        model_service.predict(image)
        measured_timings.append(_timer_ms(start))

    return {
        "cold_ms": cold_ms,
        "warmup_runs": warmups,
        "warmup_summary_ms": summarize_values(warmup_timings),
        "measured_runs": runs,
        "measured_summary_ms": summarize_values(measured_timings),
        "prediction": {
            "class_name": cold_prediction["class_name"],
            "confidence": cold_prediction["confidence"],
            "model_name": cold_prediction.get("model_name"),
            "model_version": cold_prediction.get("model_version"),
            "input_size": cold_prediction.get("input_size"),
        },
    }


def _install_db_write_timer(predict_route: Any) -> tuple[list[dict[str, Any]], Any, dict[str, str], threading.Lock]:
    original = predict_route._save_scan
    timings: list[dict[str, Any]] = []
    state = {"scenario": "unassigned"}
    lock = threading.Lock()

    def timed_save_scan(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            elapsed = _timer_ms(start)
            with lock:
                timings.append({"scenario": state["scenario"], "duration_ms": elapsed})

    predict_route._save_scan = timed_save_scan
    return timings, original, state, lock


def _restore_db_write_timer(predict_route: Any, original: Any) -> None:
    predict_route._save_scan = original


def _summarize_statuses(results: list[dict[str, Any]]) -> dict[str, int]:
    statuses: dict[str, int] = {}
    for result in results:
        key = str(result["status_code"])
        statuses[key] = statuses.get(key, 0) + 1
    return dict(sorted(statuses.items()))


def _run_api_scenario(
    *,
    client: Any,
    modules: dict[str, Any],
    scenario_name: str,
    concurrency: int,
    request_count: int,
    content: bytes,
    content_type: str,
    filename: str,
    user_id: str,
    db_timings: list[dict[str, Any]],
    scenario_state: dict[str, str],
    model_only_p50_ms: float | None,
) -> dict[str, Any]:
    scenario_state["scenario"] = scenario_name
    db_start_index = len(db_timings)
    scans_before = _scan_count(modules, user_id)
    memory_before = _memory_snapshot()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    def request_once(index: int) -> dict[str, Any]:
        start = time.perf_counter()
        response = client.post(
            "/predict",
            files={"file": (filename, content, content_type)},
        )
        latency_ms = _timer_ms(start)
        payload: dict[str, Any] | None
        try:
            payload = response.json()
        except Exception:
            payload = None
        error_detail = None
        if response.status_code != 200:
            if isinstance(payload, dict):
                error_detail = payload.get("detail")
            else:
                error_detail = response.text[:500]
        return {
            "index": index,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "ok": response.status_code == 200,
            "error_detail": error_detail,
            "class_name": payload.get("class_name") if isinstance(payload, dict) else None,
            "scan_id": payload.get("scan_id") if isinstance(payload, dict) else None,
        }

    if concurrency == 1:
        results = [request_once(index) for index in range(request_count)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(request_once, index) for index in range(request_count)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        results.sort(key=lambda item: int(item["index"]))

    wall_ms = _timer_ms(wall_start)
    cpu_seconds = time.process_time() - cpu_start
    memory_after = _memory_snapshot()
    scans_after = _scan_count(modules, user_id)
    scenario_db_timings = [
        item["duration_ms"]
        for item in db_timings[db_start_index:]
        if item["scenario"] == scenario_name
    ]
    latencies = [float(result["latency_ms"]) for result in results]
    success_latencies = [float(result["latency_ms"]) for result in results if result["ok"]]
    success_count = sum(1 for result in results if result["ok"])
    error_count = len(results) - success_count
    latency_summary = summarize_values(latencies)
    success_latency_summary = summarize_values(success_latencies)
    p50 = success_latency_summary["p50"]
    delta = (
        float(p50) - float(model_only_p50_ms)
        if p50 is not None and model_only_p50_ms is not None
        else None
    )
    return {
        "name": scenario_name,
        "concurrency": concurrency,
        "requests": request_count,
        "success_count": success_count,
        "error_count": error_count,
        "status_counts": _summarize_statuses(results),
        "wall_time_ms": wall_ms,
        "throughput_requests_per_second": float(request_count / (wall_ms / 1000.0)) if wall_ms > 0 else None,
        "process_cpu_seconds": cpu_seconds,
        "process_cpu_ms_per_request": float(cpu_seconds * 1000.0 / request_count),
        "latency_summary_ms": latency_summary,
        "success_latency_summary_ms": success_latency_summary,
        "model_only_p50_ms": model_only_p50_ms,
        "api_over_model_p50_delta_ms": delta,
        "database_write_summary_ms": summarize_values(scenario_db_timings),
        "database_write_timing_count": len(scenario_db_timings),
        "database_rows_before": scans_before,
        "database_rows_after": scans_after,
        "database_rows_inserted": scans_after - scans_before,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "errors": [
            {
                "index": result["index"],
                "status_code": result["status_code"],
                "detail": result["error_detail"],
            }
            for result in results
            if not result["ok"]
        ],
    }


def runtime_report(model_service: Any) -> dict[str, Any]:
    try:
        import onnxruntime as ort

        onnxruntime_version = ort.__version__
        onnxruntime_device = ort.get_device()
        available_providers = ort.get_available_providers()
    except Exception:
        onnxruntime_version = None
        onnxruntime_device = None
        available_providers = []
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "onnxruntime": onnxruntime_version,
        "onnxruntime_device": onnxruntime_device,
        "available_execution_providers": available_providers,
        "session_execution_providers": model_service.session.get_providers() if model_service.session else [],
        "selected_execution_provider": model_service.provider,
    }


def validate_report_schema(report: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "generated_at",
        "command",
        "config",
        "image",
        "model",
        "runtime",
        "database",
        "model_only",
        "api",
    }
    missing = required - set(report)
    if missing:
        raise ValueError(f"API benchmark report is missing top-level keys: {sorted(missing)}")
    if report["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {report['schema_version']}")
    if report["api"]["sequential"]["requests"] < 1:
        raise ValueError("Sequential benchmark must include at least one request")
    if not report["api"]["concurrency"]:
        raise ValueError("Concurrency benchmark must include at least one scenario")
    for scenario in [report["api"]["sequential"], *report["api"]["concurrency"]]:
        for key in ("latency_summary_ms", "success_count", "error_count", "database_write_summary_ms"):
            if key not in scenario:
                raise ValueError(f"Scenario {scenario.get('name')} is missing {key}")


def render_markdown(report: dict[str, Any]) -> str:
    model_only = report["model_only"]["measured_summary_ms"]
    lines = [
        "# Leaflight API Prediction Baseline",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Image: `{report['image']['path']}` ({report['image']['size_bytes']} bytes, {report['image']['content_type']})",
        (
            f"- Model: `{report['model']['model_name']} {report['model']['model_version']}` "
            f"via `{report['model']['provider']}`"
        ),
        f"- Model-only p50: `{float(model_only['p50']):.3f} ms`",
        f"- Database: `{report['database']['backend']}`",
        "",
        "## API Scenarios",
        "",
        "| Scenario | Concurrency | Requests | Success | Errors | p50 ms | p90 ms | p95 ms | Mean ms | DB write p50 ms | API minus model p50 ms | RPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    scenarios = [report["api"]["sequential"], *report["api"]["concurrency"]]
    for scenario in scenarios:
        latency = scenario["success_latency_summary_ms"]
        db_write = scenario["database_write_summary_ms"]
        lines.append(
            "| {name} | {concurrency} | {requests} | {success} | {errors} | "
            "{p50:.3f} | {p90:.3f} | {p95:.3f} | {mean:.3f} | {db_p50:.3f} | "
            "{delta:.3f} | {rps:.3f} |".format(
                name=scenario["name"],
                concurrency=scenario["concurrency"],
                requests=scenario["requests"],
                success=scenario["success_count"],
                errors=scenario["error_count"],
                p50=float(latency["p50"] or 0.0),
                p90=float(latency["p90"] or 0.0),
                p95=float(latency["p95"] or 0.0),
                mean=float(latency["mean"] or 0.0),
                db_p50=float(db_write["p50"] or 0.0),
                delta=float(scenario["api_over_model_p50_delta_ms"] or 0.0),
                rps=float(scenario["throughput_requests_per_second"] or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Python: `{report['runtime']['python']}`",
            f"- ONNX Runtime: `{report['runtime']['onnxruntime']}`",
            f"- CPU count: `{report['runtime']['cpu_count']}`",
            f"- Available execution providers: `{report['runtime']['available_execution_providers']}`",
            f"- Session execution providers: `{report['runtime']['session_execution_providers']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _unique_output_paths(output_dir: Path, created_at: datetime) -> tuple[Path, Path]:
    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"api_prediction_baseline_{stamp}.json"
    markdown_path = output_dir / f"api_prediction_baseline_{stamp}.md"
    suffix = 1
    while json_path.exists() or markdown_path.exists():
        json_path = output_dir / f"api_prediction_baseline_{stamp}_{suffix}.json"
        markdown_path = output_dir / f"api_prediction_baseline_{stamp}_{suffix}.md"
        suffix += 1
    return json_path, markdown_path


def write_outputs(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.fromisoformat(report["generated_at"])
    json_path, markdown_path = _unique_output_paths(output_dir, created_at)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _parse_concurrency_levels(value: str) -> list[int]:
    levels = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        level = int(stripped)
        if level < 1:
            raise argparse.ArgumentTypeError("Concurrency levels must be positive integers")
        levels.append(level)
    if not levels:
        raise argparse.ArgumentTypeError("At least one concurrency level is required")
    return levels


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Benchmark image does not exist: {image_path}")
    content = image_path.read_bytes()
    content_type = args.content_type or infer_content_type(image_path)
    if not content_type:
        raise ValueError("Could not infer content type; pass --content-type explicitly")

    temporary_database = None
    if args.database_url:
        database_url = args.database_url
        database_backend = "configured"
        database_path = None
    else:
        temporary_database = tempfile.TemporaryDirectory(prefix="leaflight-api-benchmark-")
        database_path = Path(temporary_database.name) / "benchmark_api.db"
        database_url = f"sqlite:///{database_path.as_posix()}"
        database_backend = "temporary_sqlite"

    try:
        _configure_benchmark_environment(database_url)
        modules = _load_backend_modules()
        TestClient = modules["TestClient"]
        app = modules["app"]
        predict_route = modules["predict_route"]
        db_timings, original_save_scan, scenario_state, _lock = _install_db_write_timer(predict_route)
        if not args.verbose_logs:
            logging.disable(logging.INFO)
        try:
            with TestClient(app) as client:
                session_info = _create_authenticated_benchmark_session(client, modules)
                accepted_image = modules["_validate_upload"](content, content_type)
                model_service = modules["model_service"]
                if not model_service.loaded:
                    raise RuntimeError(model_service.load_error or "Model service is not loaded")
                model_only = _run_model_only(
                    model_service=model_service,
                    image=accepted_image,
                    warmups=args.model_warmups,
                    runs=args.model_runs,
                )
                model_p50 = model_only["measured_summary_ms"]["p50"]
                sequential = _run_api_scenario(
                    client=client,
                    modules=modules,
                    scenario_name="sequential",
                    concurrency=1,
                    request_count=args.requests_per_level,
                    content=content,
                    content_type=content_type,
                    filename=image_path.name,
                    user_id=session_info["user_id"],
                    db_timings=db_timings,
                    scenario_state=scenario_state,
                    model_only_p50_ms=float(model_p50) if model_p50 is not None else None,
                )
                concurrency_reports = [
                    _run_api_scenario(
                        client=client,
                        modules=modules,
                        scenario_name=f"concurrency_{level}",
                        concurrency=level,
                        request_count=args.requests_per_level,
                        content=content,
                        content_type=content_type,
                        filename=image_path.name,
                        user_id=session_info["user_id"],
                        db_timings=db_timings,
                        scenario_state=scenario_state,
                        model_only_p50_ms=float(model_p50) if model_p50 is not None else None,
                    )
                    for level in args.concurrency_levels
                ]
                model_path = model_service.model_path
                report = {
                    "schema_version": SCHEMA_VERSION,
                    "generated_at": _now_utc().isoformat(),
                    "command": sys.argv,
                    "config": {
                        "requests_per_level": args.requests_per_level,
                        "concurrency_levels": args.concurrency_levels,
                        "model_warmups": args.model_warmups,
                        "model_runs": args.model_runs,
                    },
                    "image": {
                        "path": _relative_or_absolute(image_path),
                        "content_type": content_type,
                        "size_bytes": len(content),
                        "sha256": _sha256_bytes(content),
                    },
                    "database": {
                        "backend": database_backend,
                        "url_kind": "sqlite" if database_url.startswith("sqlite:///") else "configured",
                        "path": str(database_path) if database_path else None,
                        "session_user_id": session_info["user_id"],
                    },
                    "model": {
                        "path": _relative_or_absolute(model_path) if model_path else None,
                        "size_bytes": model_path.stat().st_size if model_path else None,
                        "checksum_sha256": model_service.model_checksum,
                        "architecture": model_service.architecture,
                        "model_name": model_service.model_name,
                        "model_version": model_service.model_version,
                        "input_size": model_service.input_size,
                        "class_count": len(model_service.idx_to_class),
                        "provider": model_service.provider,
                    },
                    "runtime": runtime_report(model_service),
                    "model_only": model_only,
                    "api": {
                        "sequential": sequential,
                        "concurrency": concurrency_reports,
                    },
                }
        finally:
            logging.disable(logging.NOTSET)
            _restore_db_write_timer(predict_route, original_save_scan)
        validate_report_schema(report)
        return report
    finally:
        if temporary_database is not None:
            temporary_database.cleanup()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to a valid JPEG, PNG, or WebP image.")
    parser.add_argument("--content-type", default=None, help="Override inferred upload content type.")
    parser.add_argument("--requests-per-level", type=int, default=10)
    parser.add_argument("--concurrency-levels", type=_parse_concurrency_levels, default=list(DEFAULT_CONCURRENCY_LEVELS))
    parser.add_argument("--model-warmups", type=int, default=5)
    parser.add_argument("--model-runs", type=int, default=20)
    parser.add_argument("--database-url", default=None, help="Optional explicit benchmark database URL.")
    parser.add_argument("--output-dir", default="artifacts/baselines")
    parser.add_argument("--verbose-logs", action="store_true", help="Keep application request logs visible.")
    args = parser.parse_args(argv)
    if args.requests_per_level < 1:
        parser.error("--requests-per-level must be at least 1")
    if args.model_warmups < 0:
        parser.error("--model-warmups must be non-negative")
    if args.model_runs < 1:
        parser.error("--model-runs must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(args)
    json_path, markdown_path = write_outputs(report, Path(args.output_dir))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
