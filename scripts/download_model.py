#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import re
import ssl
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.model_release import (  # noqa: E402
    ModelRelease,
    ReleaseVerificationError,
    load_release_manifest,
    verify_asset,
    verify_supporting_assets,
)


DEFAULT_RELEASE = "efficientnetv2_s_v1"
RELEASE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CHUNK_SIZE = 1024 * 1024
MAX_REDIRECTS = 5


class ModelDownloadError(RuntimeError):
    """The model could not be obtained and verified."""


@dataclass(frozen=True)
class DownloadResult:
    release: ModelRelease
    destination: Path
    size_bytes: int
    sha256: str
    reused: bool


def _open_http_response(
    url: str,
    *,
    connect_timeout: float,
    read_timeout: float,
    redirects_remaining: int = MAX_REDIRECTS,
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ModelDownloadError("Model URL must be an absolute HTTP or HTTPS URL")
    if parsed.username or parsed.password:
        raise ModelDownloadError("Model URL must not contain embedded credentials")

    connection_type = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    kwargs: dict = {"timeout": connect_timeout}
    if parsed.scheme == "https":
        kwargs["context"] = ssl.create_default_context()
    connection = connection_type(parsed.hostname, parsed.port, **kwargs)
    try:
        connection.connect()
        if connection.sock is not None:
            connection.sock.settimeout(read_timeout)
        target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        connection.request(
            "GET",
            target,
            headers={"User-Agent": "Leaflight-model-downloader/1.0", "Accept": "*/*"},
        )
        response = connection.getresponse()
        if response.status in {301, 302, 303, 307, 308}:
            location = response.getheader("Location")
            response.close()
            connection.close()
            if redirects_remaining <= 0:
                raise ModelDownloadError("Model download exceeded the redirect limit")
            if not location:
                raise ModelDownloadError("Model download redirect did not provide a location")
            redirected_url = urljoin(url, location)
            if parsed.scheme == "https" and urlsplit(redirected_url).scheme != "https":
                raise ModelDownloadError("Model download refused an HTTPS-to-HTTP redirect")
            return _open_http_response(
                redirected_url,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                redirects_remaining=redirects_remaining - 1,
            )
        if not 200 <= response.status < 300:
            status = response.status
            reason = response.reason
            response.close()
            connection.close()
            raise ModelDownloadError(f"Model download failed with HTTP {status} {reason}")
        return connection, response
    except Exception:
        connection.close()
        raise


def _download_verified(
    url: str,
    release: ModelRelease,
    destination: Path,
    *,
    connect_timeout: float,
    read_timeout: float,
) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    temp_path = Path(temp_name)
    connection: http.client.HTTPConnection | None = None
    response: http.client.HTTPResponse | None = None
    try:
        connection, response = _open_http_response(
            url, connect_timeout=connect_timeout, read_timeout=read_timeout
        )
        content_length = response.getheader("Content-Length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError as exc:
                raise ModelDownloadError("Model server returned an invalid Content-Length") from exc
            if declared_size > release.onnx.size_bytes:
                raise ModelDownloadError(
                    "Model download exceeds the maximum size allowed by the release manifest"
                )

        digest = hashlib.sha256()
        downloaded = 0
        with os.fdopen(temp_fd, "wb") as output:
            temp_fd = -1
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > release.onnx.size_bytes:
                    raise ModelDownloadError(
                        "Model download exceeds the maximum size allowed by the release manifest"
                    )
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        actual_checksum = digest.hexdigest()
        if downloaded != release.onnx.size_bytes:
            raise ModelDownloadError(
                f"Downloaded model size mismatch: expected {release.onnx.size_bytes}, got {downloaded}"
            )
        if actual_checksum != release.onnx.sha256:
            raise ModelDownloadError(
                "Downloaded model SHA-256 mismatch: "
                f"expected {release.onnx.sha256}, got {actual_checksum}"
            )
        os.replace(temp_path, destination)
        return downloaded, actual_checksum
    except ModelDownloadError:
        raise
    except Exception as exc:
        raise ModelDownloadError(f"Model download failed: {exc}") from exc
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        if response is not None:
            response.close()
        if connection is not None:
            connection.close()
        temp_path.unlink(missing_ok=True)


def install_release(
    manifest_path: str | Path,
    *,
    url: str | None = None,
    verify_only: bool = False,
    connect_timeout: float = 10.0,
    read_timeout: float = 60.0,
    environment: dict[str, str] | None = None,
) -> DownloadResult:
    if connect_timeout <= 0 or read_timeout <= 0:
        raise ModelDownloadError("Connection and read timeouts must be positive")
    release = load_release_manifest(manifest_path)
    verify_supporting_assets(release)
    destination = release.path_for(release.onnx)

    existing_error: ReleaseVerificationError | None = None
    if destination.exists():
        try:
            size_bytes, checksum = verify_asset(destination, release.onnx)
            return DownloadResult(release, destination, size_bytes, checksum, reused=True)
        except ReleaseVerificationError as exc:
            existing_error = exc

    if verify_only:
        if existing_error is not None:
            raise ModelDownloadError(f"Existing model verification failed: {existing_error}")
        raise ModelDownloadError(f"Model is missing: {destination}")

    environment = os.environ if environment is None else environment
    configured_url = (
        url
        or environment.get(release.download_url_environment_variable)
        or release.download_url
    )
    if not configured_url:
        prefix = f"Existing model verification failed: {existing_error}. " if existing_error else ""
        raise ModelDownloadError(
            prefix
            + "No model download URL is configured. Set "
            + release.download_url_environment_variable
            + " or pass --url <release-url>, then run python scripts/download_model.py."
        )

    size_bytes, checksum = _download_verified(
        configured_url,
        release,
        destination,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
    )
    return DownloadResult(release, destination, size_bytes, checksum, reused=False)


def _manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest:
        return Path(args.manifest)
    if RELEASE_NAME_PATTERN.fullmatch(args.release) is None:
        raise ModelDownloadError("Release name contains unsupported characters")
    return PROJECT_ROOT / "models" / "releases" / args.release / "release.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and SHA-256 verify the active Leaflight ONNX release."
    )
    parser.add_argument("--release", default=DEFAULT_RELEASE, help="Versioned release directory name")
    parser.add_argument("--manifest", help="Explicit release.json path (primarily for release validation)")
    parser.add_argument("--url", help="Versioned ONNX release URL; overrides the manifest's environment variable")
    parser.add_argument("--verify-only", action="store_true", help="Verify the local release without downloading")
    parser.add_argument("--connect-timeout", type=float, default=10.0, help="Connection timeout in seconds")
    parser.add_argument("--read-timeout", type=float, default=60.0, help="Socket read timeout in seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = install_release(
            _manifest_path(args),
            url=args.url,
            verify_only=args.verify_only,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
        )
    except (FileNotFoundError, ValueError, ModelDownloadError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    action = "verified existing model" if result.reused else "downloaded and verified model"
    print(f"Leaflight {action}")
    print(f"Version: {result.release.version}")
    print(f"Destination: {result.destination}")
    print(f"File size: {result.size_bytes} bytes")
    print(f"SHA-256: {result.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
