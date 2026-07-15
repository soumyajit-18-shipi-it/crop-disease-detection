from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from backend.api.model_loader import ModelService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_RELEASE = (
    PROJECT_ROOT / "models" / "releases" / "efficientnetv2_s_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_backend_startup_fails_closed_when_model_is_missing(tmp_path):
    service = ModelService(
        tmp_path / "model.onnx",
        tmp_path / "model.json",
        tmp_path / "release.json",
    )

    assert service.load() is False
    assert service.loaded is False
    assert "ONNX model does not exist" in service.load_error
    assert "python scripts/download_model.py" in service.load_error


def test_backend_startup_fails_closed_when_model_checksum_is_invalid(tmp_path):
    release_dir = tmp_path / "test_release"
    release_dir.mkdir()
    metadata_path = release_dir / "model.json"
    metrics_path = release_dir / "metrics.json"
    model_path = release_dir / "model.onnx"
    shutil.copyfile(PRODUCTION_RELEASE / "model.json", metadata_path)
    shutil.copyfile(PRODUCTION_RELEASE / "metrics.json", metrics_path)
    model_path.write_bytes(b"corrupted-onnx")

    manifest = json.loads(
        (PRODUCTION_RELEASE / "release.json").read_text(encoding="utf-8")
    )
    manifest["release_name"] = "test_release"
    manifest["onnx"]["size_bytes"] = len(b"corrupted-onnx")
    manifest["onnx"]["sha256"] = hashlib.sha256(b"different-onnx").hexdigest()
    manifest["metadata"]["size_bytes"] = metadata_path.stat().st_size
    manifest["metadata"]["sha256"] = _sha256(metadata_path)
    manifest["metrics"]["size_bytes"] = metrics_path.stat().st_size
    manifest["metrics"]["sha256"] = _sha256(metrics_path)
    manifest_path = release_dir / "release.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (release_dir / "checksum.sha256").write_text(
        f"{manifest['onnx']['sha256']}  model.onnx\n"
        f"{manifest['metadata']['sha256']}  model.json\n"
        f"{manifest['metrics']['sha256']}  metrics.json\n",
        encoding="utf-8",
    )

    service = ModelService(model_path, metadata_path, manifest_path)

    assert service.load() is False
    assert service.loaded is False
    assert "SHA-256 mismatch" in service.load_error
    assert "python scripts/download_model.py" in service.load_error
