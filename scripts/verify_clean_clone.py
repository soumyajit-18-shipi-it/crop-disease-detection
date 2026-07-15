#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_RELATIVE = Path("models/releases/efficientnetv2_s_v1")


def _copy_git_visible_workspace(destination: Path) -> None:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    for raw_name in result.stdout.split(b"\0"):
        if not raw_name:
            continue
        relative = Path(os.fsdecode(raw_name))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Refusing unsafe Git-visible path: {relative}")
        source = PROJECT_ROOT / relative
        if not source.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_json(url: str, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                return response.status, json.loads(response.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def _prediction_body() -> tuple[bytes, str]:
    from PIL import Image

    image = Image.effect_noise((360, 240), 40).convert("RGB")
    image_buffer = BytesIO()
    image.save(image_buffer, format="JPEG")
    boundary = f"leaflight-{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="clean-clone-leaf.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + image_buffer.getvalue() + f"\r\n--{boundary}--\r\n".encode()
    return body, boundary


def _post_prediction(url: str, session_token: str, csrf_token: str) -> dict:
    body, boundary = _prediction_body()
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Cookie": f"leaflight_session={session_token}; leaflight_csrf={csrf_token}",
            "X-CSRF-Token": csrf_token,
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"POST /predict returned HTTP {response.status}")
        payload = response.read()
        if not payload:
            raise RuntimeError("POST /predict returned an empty HTTP 200 response")
        return json.loads(payload)


def _create_test_session(staging: Path, environment: dict[str, str]) -> tuple[str, str]:
    setup = """
import json
from backend.api.auth import create_session, isoformat, utc_now
from backend.db.database import connect_database
from backend.db.seed_disease_data import seed_database

seed_database()
now = isoformat(utc_now())
with connect_database() as connection:
    connection.execute(
        \"\"\"INSERT INTO users(
            id, name, email, auth_provider, provider_account_id, created_at, last_login_at
        ) VALUES ('clean-clone-user', 'Clean Clone Test', 'clean-clone@example.test',
                  'google', 'clean-clone-google', ?, ?)\"\"\",
        (now, now),
    )
    connection.commit()
token, csrf, _ = create_session('clean-clone-user')
print(json.dumps({'session': token, 'csrf': csrf}))
"""
    result = subprocess.run(
        [sys.executable, "-c", setup],
        cwd=staging,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout.strip())
    return str(payload["session"]), str(payload["csrf"])


def main() -> int:
    source_model = PROJECT_ROOT / RELEASE_RELATIVE / "model.onnx"
    if not source_model.is_file():
        print(
            "ERROR: The local clean-clone simulation needs the existing verified production "
            "model as its loopback HTTP source.",
            file=sys.stderr,
        )
        return 1

    payload_size = source_model.stat().st_size

    class ModelHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path != "/model.onnx":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(payload_size))
            self.end_headers()
            with source_model.open("rb") as model_file:
                shutil.copyfileobj(model_file, self.wfile, length=1024 * 1024)

        def log_message(self, _format, *_args):
            return

    model_server = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
    model_thread = threading.Thread(target=model_server.serve_forever, daemon=True)
    model_thread.start()
    api_process: subprocess.Popen | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix="leaflight-clean-clone-",
            ignore_cleanup_errors=(os.name == "nt"),
        ) as temporary:
            staging = Path(temporary) / "crop-disease-detection"
            staging.mkdir()
            _copy_git_visible_workspace(staging)
            staged_model = staging / RELEASE_RELATIVE / "model.onnx"
            if staged_model.exists():
                raise RuntimeError("Ignored model.onnx unexpectedly appeared in clean staging")

            model_url = f"http://127.0.0.1:{model_server.server_port}/model.onnx"
            download = subprocess.run(
                [sys.executable, "scripts/download_model.py", "--url", model_url],
                cwd=staging,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [sys.executable, "scripts/download_model.py", "--verify-only"],
                cwd=staging,
                check=True,
                capture_output=True,
                text=True,
            )

            api_port = _available_port()
            environment = os.environ.copy()
            environment.update(
                {
                    "DB_PATH": str(staging / "runtime" / "disease_info.db"),
                    "LOG_DIR": str(staging / "runtime" / "logs"),
                    "LOG_TO_FILE": "false",
                    "PORT": str(api_port),
                    "AUTH_SECRET": "clean-clone-only-secret-with-at-least-32-characters",
                    "COOKIE_SECURE": "false",
                }
            )
            api_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "backend.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(api_port),
                ],
                cwd=staging,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            base_url = f"http://127.0.0.1:{api_port}"
            health_status, health = _wait_for_json(f"{base_url}/health")
            classes_status, classes = _wait_for_json(f"{base_url}/classes")
            session_token, csrf_token = _create_test_session(staging, environment)
            prediction = _post_prediction(f"{base_url}/predict", session_token, csrf_token)

            if health_status != 200 or health.get("model_loaded") is not True:
                raise RuntimeError(f"Unexpected health response: {health}")
            if classes_status != 200 or len(classes) != 15:
                raise RuntimeError(f"Unexpected classes response: {classes}")
            if prediction.get("mock") is not False or prediction.get("model_version") != "v1":
                raise RuntimeError(f"Unexpected prediction response: {prediction}")

            api_process.terminate()
            try:
                api_process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                api_process.kill()
                api_process.communicate(timeout=5)
            api_process = None

            print("Clean-clone local HTTP simulation passed")
            print(download.stdout.strip())
            print("GET /health: 200 (model_loaded=true)")
            print("GET /classes: 200 (15 classes)")
            print("POST /predict: 200 (authenticated real ONNX v1 prediction)")
            return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: clean-clone simulation failed: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr.strip(), file=sys.stderr)
        if api_process is not None:
            if api_process.poll() is None:
                api_process.terminate()
            _stdout, stderr = api_process.communicate(timeout=10)
            if stderr:
                print(stderr.strip(), file=sys.stderr)
            api_process = None
        return 1
    finally:
        if api_process is not None and api_process.poll() is None:
            api_process.terminate()
            try:
                api_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                api_process.kill()
                api_process.wait(timeout=5)
        model_server.shutdown()
        model_server.server_close()
        model_thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
