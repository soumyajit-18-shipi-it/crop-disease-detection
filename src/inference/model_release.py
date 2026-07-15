from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ReleaseManifestError(ValueError):
    """The release manifest is missing or violates the release schema."""


class ReleaseVerificationError(ValueError):
    """A release asset does not match its immutable manifest entry."""


@dataclass(frozen=True)
class ReleaseAsset:
    filename: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ModelRelease:
    manifest_path: Path
    release_name: str
    architecture: str
    version: str
    input_width: int
    input_height: int
    input_channels: int
    input_color_space: str
    class_count: int
    onnx: ReleaseAsset
    metadata: ReleaseAsset
    metrics: ReleaseAsset
    checksum_filename: str
    minimum_backend_version: str
    download_url_environment_variable: str
    download_url: str | None
    created_at: str
    validation_status: str
    onnx_parity_status: str

    @property
    def release_dir(self) -> Path:
        return self.manifest_path.parent

    def path_for(self, asset: ReleaseAsset) -> Path:
        return self.release_dir / asset.filename


def _required_mapping(payload: Mapping, key: str) -> Mapping:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ReleaseManifestError(f"Release manifest field '{key}' must be an object")
    return value


def _required_string(payload: Mapping, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReleaseManifestError(
            f"Release manifest field '{key}' must be a non-empty string"
        )
    return value.strip()


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ReleaseManifestError(
            f"Release manifest field '{field_name}' must be a positive integer"
        )
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ReleaseManifestError(
            f"Release manifest field '{field_name}' must be a positive integer"
        ) from exc
    if converted <= 0:
        raise ReleaseManifestError(
            f"Release manifest field '{field_name}' must be a positive integer"
        )
    return converted


def _safe_filename(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReleaseManifestError(
            f"Release manifest field '{field_name}' must be a non-empty filename"
        )
    filename = value.strip()
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ReleaseManifestError(
            f"Release manifest field '{field_name}' must not contain a path"
        )
    return filename


def _asset(payload: Mapping, key: str) -> ReleaseAsset:
    value = _required_mapping(payload, key)
    filename = _safe_filename(value.get("filename"), f"{key}.filename")
    checksum = _required_string(value, "sha256").lower()
    if SHA256_PATTERN.fullmatch(checksum) is None:
        raise ReleaseManifestError(
            f"Release manifest field '{key}.sha256' must be 64 lowercase hexadecimal characters"
        )
    size_bytes = _positive_integer(value.get("size_bytes"), f"{key}.size_bytes")
    return ReleaseAsset(filename=filename, sha256=checksum, size_bytes=size_bytes)


def load_release_manifest(path: str | Path) -> ModelRelease:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Release manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseManifestError(
            f"Release manifest is not valid JSON: {manifest_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ReleaseManifestError("Release manifest root must be an object")
    if payload.get("schema_version") != 1:
        raise ReleaseManifestError("Unsupported release manifest schema_version")

    input_spec = _required_mapping(payload, "input")
    raw_download_url = payload.get("download_url")
    if raw_download_url is not None and (
        not isinstance(raw_download_url, str) or not raw_download_url.strip()
    ):
        raise ReleaseManifestError(
            "Release manifest field 'download_url' must be a non-empty string when present"
        )

    release = ModelRelease(
        manifest_path=manifest_path,
        release_name=_required_string(payload, "release_name"),
        architecture=_required_string(payload, "architecture"),
        version=_required_string(payload, "version"),
        input_width=_positive_integer(input_spec.get("width"), "input.width"),
        input_height=_positive_integer(input_spec.get("height"), "input.height"),
        input_channels=_positive_integer(input_spec.get("channels"), "input.channels"),
        input_color_space=_required_string(input_spec, "color_space").upper(),
        class_count=_positive_integer(payload.get("class_count"), "class_count"),
        onnx=_asset(payload, "onnx"),
        metadata=_asset(payload, "metadata"),
        metrics=_asset(payload, "metrics"),
        checksum_filename=_safe_filename(
            payload.get("checksum_filename"), "checksum_filename"
        ),
        minimum_backend_version=_required_string(payload, "minimum_backend_version"),
        download_url_environment_variable=_required_string(
            payload, "download_url_environment_variable"
        ),
        download_url=raw_download_url.strip() if raw_download_url is not None else None,
        created_at=_required_string(payload, "created_at"),
        validation_status=_required_string(payload, "validation_status"),
        onnx_parity_status=_required_string(payload, "onnx_parity_status"),
    )
    filenames = {
        release.onnx.filename,
        release.metadata.filename,
        release.metrics.filename,
        release.checksum_filename,
        manifest_path.name,
    }
    if len(filenames) != 5:
        raise ReleaseManifestError("Release asset filenames must be unique")
    return release


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path: str | Path, asset: ReleaseAsset) -> tuple[int, str]:
    asset_path = Path(path)
    if not asset_path.is_file():
        raise ReleaseVerificationError(f"Required release asset is missing: {asset_path}")
    if asset_path.is_symlink():
        raise ReleaseVerificationError(f"Release assets must not be symbolic links: {asset_path}")
    actual_size = asset_path.stat().st_size
    if actual_size != asset.size_bytes:
        raise ReleaseVerificationError(
            f"Size mismatch for {asset_path}: expected {asset.size_bytes}, got {actual_size}"
        )
    actual_checksum = sha256_file(asset_path)
    if actual_checksum != asset.sha256:
        raise ReleaseVerificationError(
            f"SHA-256 mismatch for {asset_path}: expected {asset.sha256}, got {actual_checksum}"
        )
    return actual_size, actual_checksum


def verify_supporting_assets(release: ModelRelease) -> None:
    verify_asset(release.path_for(release.metadata), release.metadata)
    verify_asset(release.path_for(release.metrics), release.metrics)
    checksum_path = release.release_dir / release.checksum_filename
    if not checksum_path.is_file():
        raise ReleaseVerificationError(
            f"Required release checksum file is missing: {checksum_path}"
        )
    entries: dict[str, str] = {}
    try:
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            checksum, filename = line.split(maxsplit=1)
            filename = filename.lstrip(" *")
            if filename in entries:
                raise ValueError(f"Duplicate checksum entry: {filename}")
            entries[filename] = checksum.lower()
    except (OSError, ValueError) as exc:
        raise ReleaseVerificationError(
            f"Release checksum file is malformed: {checksum_path}"
        ) from exc
    expected = {
        release.onnx.filename: release.onnx.sha256,
        release.metadata.filename: release.metadata.sha256,
        release.metrics.filename: release.metrics.sha256,
    }
    if entries != expected:
        raise ReleaseVerificationError(
            f"Release checksum file does not match {release.manifest_path}"
        )
