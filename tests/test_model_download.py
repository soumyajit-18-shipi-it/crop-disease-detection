from __future__ import annotations

import hashlib
import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts.download_model import ModelDownloadError, install_release
from src.inference.model_release import load_release_manifest


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_release(
    tmp_path: Path,
    expected_model: bytes,
    *,
    existing_model: bytes | None = None,
) -> tuple[Path, Path]:
    release_dir = tmp_path / "models" / "releases" / "test_release_v1"
    release_dir.mkdir(parents=True)
    metadata = b'{"model_version":"v1"}\n'
    metrics = b'{"accuracy":1.0}\n'
    (release_dir / "model.json").write_bytes(metadata)
    (release_dir / "metrics.json").write_bytes(metrics)
    if existing_model is not None:
        (release_dir / "model.onnx").write_bytes(existing_model)

    model_checksum = _sha256(expected_model)
    metadata_checksum = _sha256(metadata)
    metrics_checksum = _sha256(metrics)
    (release_dir / "checksum.sha256").write_text(
        f"{model_checksum}  model.onnx\n"
        f"{metadata_checksum}  model.json\n"
        f"{metrics_checksum}  metrics.json\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "release_name": "test_release_v1",
        "architecture": "test_architecture",
        "version": "v1",
        "input": {"width": 8, "height": 8, "channels": 3, "color_space": "RGB"},
        "class_count": 2,
        "onnx": {
            "filename": "model.onnx",
            "sha256": model_checksum,
            "size_bytes": len(expected_model),
        },
        "metadata": {
            "filename": "model.json",
            "sha256": metadata_checksum,
            "size_bytes": len(metadata),
        },
        "metrics": {
            "filename": "metrics.json",
            "sha256": metrics_checksum,
            "size_bytes": len(metrics),
        },
        "checksum_filename": "checksum.sha256",
        "minimum_backend_version": "1.0.0",
        "download_url_environment_variable": "LEAFLIGHT_MODEL_URL",
        "created_at": "2026-07-15T00:00:00+00:00",
        "validation_status": "passed",
        "onnx_parity_status": "passed",
    }
    manifest_path = release_dir / "release.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, release_dir


@contextmanager
def _model_server(payload: bytes, *, interrupt: bool = False):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if interrupt:
                self.wfile.write(payload[: max(1, len(payload) // 2)])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            self.wfile.write(payload)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/model.onnx"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_valid_checksum_succeeds(tmp_path):
    payload = b"valid-onnx-payload" * 128
    manifest_path, release_dir = _write_release(tmp_path, payload)
    with _model_server(payload) as url:
        result = install_release(manifest_path, url=url)

    assert result.reused is False
    assert result.sha256 == _sha256(payload)
    assert (release_dir / "model.onnx").read_bytes() == payload


def test_invalid_checksum_fails(tmp_path):
    expected = b"expected-model-bytes"
    downloaded = b"tampered-model-bytes"
    assert len(downloaded) == len(expected)
    manifest_path, release_dir = _write_release(tmp_path, expected)
    with _model_server(downloaded) as url:
        with pytest.raises(ModelDownloadError, match="SHA-256 mismatch"):
            install_release(manifest_path, url=url)

    assert not (release_dir / "model.onnx").exists()


def test_existing_valid_model_is_reused_without_network(tmp_path):
    payload = b"already-present-and-valid"
    manifest_path, release_dir = _write_release(
        tmp_path, payload, existing_model=payload
    )

    result = install_release(manifest_path, url="http://127.0.0.1:1/unreachable")

    assert result.reused is True
    assert (release_dir / "model.onnx").read_bytes() == payload


def test_existing_corrupted_model_is_rejected(tmp_path):
    expected = b"expected-model"
    manifest_path, _ = _write_release(
        tmp_path, expected, existing_model=b"corrupt-model!"
    )

    with pytest.raises(ModelDownloadError, match="Existing model verification failed"):
        install_release(manifest_path, verify_only=True)


def test_missing_url_has_actionable_error(tmp_path):
    manifest_path, _ = _write_release(tmp_path, b"expected-model")

    with pytest.raises(ModelDownloadError) as error:
        install_release(manifest_path, environment={})

    message = str(error.value)
    assert "LEAFLIGHT_MODEL_URL" in message
    assert "--url" in message
    assert "python scripts/download_model.py" in message


def test_manifest_download_url_is_used_by_default(tmp_path):
    payload = b"manifest-url-model" * 64
    manifest_path, release_dir = _write_release(tmp_path, payload)
    with _model_server(payload) as url:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["download_url"] = url
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result = install_release(manifest_path, environment={})

    assert result.reused is False
    assert (release_dir / "model.onnx").read_bytes() == payload


def test_interrupted_download_leaves_no_final_model(tmp_path):
    payload = b"interrupted-model-payload" * 256
    manifest_path, release_dir = _write_release(tmp_path, payload)
    with _model_server(payload, interrupt=True) as url:
        with pytest.raises(ModelDownloadError, match="size mismatch"):
            install_release(manifest_path, url=url)

    assert not (release_dir / "model.onnx").exists()
    assert list(release_dir.glob("*.part")) == []


def test_temporary_file_is_cleaned_after_checksum_failure(tmp_path):
    expected = b"expected-model-bytes"
    downloaded = b"tampered-model-bytes"
    manifest_path, release_dir = _write_release(tmp_path, expected)
    with _model_server(downloaded) as url:
        with pytest.raises(ModelDownloadError):
            install_release(manifest_path, url=url)

    assert list(release_dir.glob("*.part")) == []


def test_release_manifest_parsing(tmp_path):
    payload = b"manifest-model"
    manifest_path, release_dir = _write_release(tmp_path, payload)

    release = load_release_manifest(manifest_path)

    assert release.release_name == "test_release_v1"
    assert release.version == "v1"
    assert release.class_count == 2
    assert release.path_for(release.onnx) == release_dir / "model.onnx"
    assert release.onnx.sha256 == _sha256(payload)
