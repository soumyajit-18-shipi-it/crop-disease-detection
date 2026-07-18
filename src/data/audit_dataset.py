from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

SCHEMA_VERSION = "1.0"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_OUTPUT_DIR = Path("artifacts/baselines/dataset_audit")


@dataclass
class ImageRecord:
    path: Path
    split: str | None = None
    class_name: str | None = None
    size_bytes: int = 0
    width: int = 0
    height: int = 0
    color_mode: str | None = None
    exact_hash: str | None = None
    perceptual_hash: str | None = None
    decode_error: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute a difference hash (dhash) as a hex string."""
    gray = image.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(gray, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    hex_length = (len(bits) + 3) // 4
    return "{:0{}x}".format(
        int("".join("1" if b else "0" for b in bits), 2), hex_length
    )


def _hamming_distance(hex_a: str, hex_b: str) -> int:
    val_a = int(hex_a, 16)
    val_b = int(hex_b, 16)
    return (val_a ^ val_b).bit_count()


def _analyze_image(path: Path) -> ImageRecord:
    size_bytes = path.stat().st_size
    try:
        with Image.open(path) as img:
            width, height = img.size
            color_mode = img.mode
            exact_hash = _sha256_file(path)
            try:
                p_hash = _dhash(img)
            except Exception:
                p_hash = None
        return ImageRecord(
            path=path,
            size_bytes=size_bytes,
            width=width,
            height=height,
            color_mode=color_mode,
            exact_hash=exact_hash,
            perceptual_hash=p_hash,
        )
    except Exception as exc:
        return ImageRecord(
            path=path, size_bytes=size_bytes, decode_error=str(exc)
        )


def _collect_split_images(
    root_dir: Path,
) -> tuple[dict[str, list[ImageRecord]], dict[str, dict[str, list[ImageRecord]]]]:
    split_records: dict[str, list[ImageRecord]] = defaultdict(list)
    by_class_split: dict[str, dict[str, list[ImageRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    if not root_dir.exists():
        return split_records, by_class_split

    for split in ("train", "val", "test"):
        split_dir = root_dir / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            class_name = class_dir.name
            class_records: list[ImageRecord] = []
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    record = _analyze_image(path)
                    record.split = split
                    record.class_name = class_name
                    class_records.append(record)
            split_records[split].extend(class_records)
            by_class_split[split][class_name] = class_records
    return split_records, by_class_split


def _collect_raw_images(
    root_dir: Path,
) -> tuple[list[ImageRecord], dict[str, list[ImageRecord]]]:
    records: list[ImageRecord] = []
    by_class: dict[str, list[ImageRecord]] = defaultdict(list)
    if not root_dir.exists():
        return records, by_class
    for class_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        class_name = class_dir.name
        class_records: list[ImageRecord] = []
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                record = _analyze_image(path)
                record.class_name = class_name
                class_records.append(record)
        records.extend(class_records)
        by_class[class_name] = class_records
    return records, by_class


def _find_exact_duplicates(
    records: list[ImageRecord],
) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.exact_hash:
            groups[record.exact_hash].append(str(record.path.resolve().as_posix()))
    return [
        {"hash": h, "count": len(paths), "paths": sorted(paths)}
        for h, paths in groups.items()
        if len(paths) > 1
    ]


def _find_near_duplicates(
    records: list[ImageRecord],
    hamming_threshold: int = 10,
    max_comparisons: int = 500_000,
) -> list[dict[str, Any]]:
    valid = [r for r in records if r.perceptual_hash]
    if len(valid) < 2:
        return []

    prefix_len = max(1, min(4, len(valid[0].perceptual_hash) // 2))
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(valid):
        prefix = record.perceptual_hash[:prefix_len]
        buckets[prefix].append(idx)

    seen = set()
    pairs: list[dict[str, Any]] = []
    total_compared = 0

    for indices in buckets.values():
        if len(indices) < 2:
            continue
        for i_idx in range(len(indices)):
            for j_idx in range(i_idx + 1, len(indices)):
                if total_compared >= max_comparisons:
                    break
                a = valid[indices[i_idx]]
                b = valid[indices[j_idx]]
                dist = _hamming_distance(a.perceptual_hash, b.perceptual_hash)
                if dist <= hamming_threshold:
                    pair_key = (
                        str(a.path.resolve().as_posix()),
                        str(b.path.resolve().as_posix()),
                    )
                    if pair_key not in seen:
                        seen.add(pair_key)
                        pairs.append(
                            {
                                "path_a": pair_key[0],
                                "path_b": pair_key[1],
                                "hamming_distance": dist,
                            }
                        )
                total_compared += 1
            if total_compared >= max_comparisons:
                break
        if total_compared >= max_comparisons:
            break

    return pairs


def _class_label_conflicts(
    class_names: list[str], threshold: float = 0.82
) -> list[dict[str, Any]]:
    conflicts = []
    for i, left in enumerate(class_names):
        for right in class_names[i + 1 :]:
            ratio = SequenceMatcher(None, left, right).ratio()
            if ratio >= threshold:
                conflicts.append(
                    {
                        "class_a": left,
                        "class_b": right,
                        "similarity": round(ratio, 4),
                    }
                )
    return conflicts


def _class_imbalance_ratios(
    counts: dict[str, int],
) -> dict[str, Any]:
    if not counts:
        return {
            "max_class_share": 0.0,
            "min_class_share": 0.0,
            "imbalance_ratio": 0.0,
        }
    values = list(counts.values())
    total = sum(values)
    max_count = max(values)
    min_count = min(values)
    return {
        "max_class_share": max_count / total if total else 0.0,
        "min_class_share": min_count / total if total else 0.0,
        "imbalance_ratio": max_count / min_count if min_count else 0.0,
    }


def _detect_split_leakage(
    split_records: dict[str, list[ImageRecord]],
) -> list[dict[str, Any]]:
    split_hashes = {
        name: {r.exact_hash for r in records if r.exact_hash}
        for name, records in split_records.items()
    }
    leakage = []
    for split_a in split_hashes:
        for split_b in split_hashes:
            if split_a >= split_b:
                continue
            overlap = split_hashes[split_a] & split_hashes[split_b]
            if overlap:
                leakage.append(
                    {
                        "split_a": split_a,
                        "split_b": split_b,
                        "overlapping_hashes": len(overlap),
                    }
                )
    return leakage


def _read_split_manifest(
    manifest_path: Path,
) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return payload
    except Exception:
        return None


def audit_dataset(
    data_root: str | Path = "data",
    processed_dir: str | Path = "data/processed",
    raw_dir: str | Path = "data/raw",
    split_manifest: str | Path | None = "data/splits/phase1_split.json",
    class_mapping_path: str | Path = "data/class_mapping.json",
) -> dict[str, Any]:
    """Audit the Leaflight dataset and return a structured report."""
    data_root_path = Path(data_root)
    processed_path = Path(processed_dir)
    raw_path = Path(raw_dir)
    mapping_path = Path(class_mapping_path)

    class_to_idx: dict[str, int] = {}
    if mapping_path.exists():
        try:
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))
            class_to_idx = {
                str(k): int(v) for k, v in payload.get("class_to_idx", {}).items()
            }
        except Exception:
            class_to_idx = {}

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp,
        "dataset_root": str(data_root_path.resolve()),
        "processed_dir": str(processed_path.resolve()),
        "raw_dir": str(raw_path.resolve()),
        "class_count": len(class_to_idx),
        "classes": sorted(class_to_idx.keys()),
        "splits": {},
        "class_distribution": {},
        "raw": {},
        "manifest": {},
        "issues": {
            "decode_failures": [],
            "zero_byte_files": [],
            "missing_source_files": [],
            "exact_duplicate_groups": [],
            "near_duplicate_groups": [],
            "class_label_conflicts": [],
            "split_leakage": [],
        },
        "statistics": {},
    }

    split_records, by_class_split = _collect_split_images(processed_path)

    for split_name in ("train", "val", "test"):
        records = split_records.get(split_name, [])
        report["splits"][split_name] = len(records)
        class_counts: dict[str, int] = Counter(
            r.class_name for r in records if r.class_name
        )
        report["class_distribution"][split_name] = dict(sorted(class_counts.items()))

        decode_failures = [r for r in records if r.decode_error]
        zero_byte = [str(r.path.resolve().as_posix()) for r in records if r.size_bytes == 0]
        report["issues"]["decode_failures"].extend(
            [
                {
                    "path": str(r.path.resolve().as_posix()),
                    "split": split_name,
                    "class": r.class_name,
                    "error": r.decode_error,
                }
                for r in decode_failures
            ]
        )
        report["issues"]["zero_byte_files"].extend(zero_byte)

    report["issues"]["exact_duplicate_groups"] = _find_exact_duplicates(
        [r for records in split_records.values() for r in records]
    )

    report["issues"]["near_duplicate_groups"] = _find_near_duplicates(
        [r for records in split_records.values() for r in records]
    )

    all_classes = sorted(
        {r.class_name for records in split_records.values() for r in records if r.class_name}
    )
    if all_classes:
        report["issues"]["class_label_conflicts"] = _class_label_conflicts(all_classes)

    report["issues"]["split_leakage"] = _detect_split_leakage(split_records)

    all_counts = Counter(
        r.class_name
        for records in split_records.values()
        for r in records
        if r.class_name
    )
    report["statistics"]["class_imbalance"] = _class_imbalance_ratios(dict(all_counts))

    total = sum(report["splits"].values())
    report["statistics"]["total_images"] = total
    report["statistics"]["train_ratio"] = report["splits"].get("train", 0) / total if total else 0.0
    report["statistics"]["val_ratio"] = report["splits"].get("val", 0) / total if total else 0.0
    report["statistics"]["test_ratio"] = report["splits"].get("test", 0) / total if total else 0.0

    raw_records, raw_by_class = _collect_raw_images(raw_path)
    raw_class_counts = {k: len(v) for k, v in raw_by_class.items()}
    report["raw"]["total_images"] = len(raw_records)
    report["raw"]["class_distribution"] = dict(sorted(raw_class_counts.items()))
    report["raw"]["decode_failures"] = sum(1 for r in raw_records if r.decode_error)
    report["raw"]["zero_byte_files"] = sum(1 for r in raw_records if r.size_bytes == 0)

    manifest = _read_split_manifest(Path(split_manifest)) if split_manifest else None
    if manifest:
        report["manifest"]["path"] = str(Path(split_manifest).resolve())
        report["manifest"]["schema_version"] = manifest.get("schema_version")
        report["manifest"]["seed"] = manifest.get("seed")
        report["manifest"]["ratios"] = manifest.get("ratios")
        report["manifest"]["skipped_optional_datasets"] = manifest.get(
            "skipped_optional_datasets", []
        )
        items = manifest.get("items", [])
        manifest_hashes = {
            item.get("sha256") or item.get("sample_id")
            for item in items
            if item.get("sha256") or item.get("sample_id")
        }
        file_hashes = {
            r.exact_hash
            for records in split_records.values()
            for r in records
            if r.exact_hash
        }
        missing = manifest_hashes - file_hashes
        report["manifest"]["missing_source_files"] = len(missing)

    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Audit Report",
        "",
        f"**Generated:** {report['timestamp']}",
        f"**Schema:** `{report['schema_version']}`",
        f"**Dataset root:** `{report['dataset_root']}`",
        "",
        "## Summary",
        "",
        f"- Classes: `{report['class_count']}`",
        f"- Total processed images: `{report['statistics']['total_images']}`",
        f"- Raw images: `{report['raw']['total_images']}`",
        "",
    ]

    lines.append("## Split Sizes")
    lines.append("")
    lines.append("| Split | Images | Ratio |")
    lines.append("| --- | ---: | ---: |")
    for split in ("train", "val", "test"):
        count = report["splits"].get(split, 0)
        ratio_key = f"{split}_ratio"
        ratio = report["statistics"].get(ratio_key, 0.0)
        lines.append(f"| {split} | {count} | {ratio*100:.1f}% |")
    lines.append("")

    lines.append("## Class Distribution")
    lines.append("")
    for split in ("train", "val", "test"):
        dist = report["class_distribution"].get(split)
        if not dist:
            continue
        lines.append(f"### {split.upper()}")
        lines.append("")
        lines.append("| Class | Count |")
        lines.append("| --- | ---: |")
        for cls, count in sorted(dist.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"| {cls} | {count} |")
        lines.append("")

    imbalance = report["statistics"].get("class_imbalance", {})
    if imbalance:
        lines.append("## Class Imbalance")
        lines.append("")
        lines.append(
            f"- Max class share: `{imbalance.get('max_class_share', 0):.2%}`"
        )
        lines.append(
            f"- Min class share: `{imbalance.get('min_class_share', 0):.2%}`"
        )
        lines.append(
            f"- Imbalance ratio (max/min): `{imbalance.get('imbalance_ratio', 0):.2f}`"
        )
        lines.append("")

    if report["issues"]["decode_failures"]:
        lines.append("## Decode Failures")
        lines.append("")
        for item in report["issues"]["decode_failures"][:20]:
            lines.append(
                f"- `{item['path']}` ({item['split']}/{item['class']}): {item['error']}"
            )
        if len(report["issues"]["decode_failures"]) > 20:
            lines.append(
                f"- ... and {len(report['issues']['decode_failures']) - 20} more"
            )
        lines.append("")

    if report["issues"]["zero_byte_files"]:
        lines.append("## Zero-Byte Files")
        lines.append("")
        for path in report["issues"]["zero_byte_files"][:20]:
            lines.append(f"- `{path}`")
        if len(report["issues"]["zero_byte_files"]) > 20:
            lines.append(
                f"- ... and {len(report['issues']['zero_byte_files']) - 20} more"
            )
        lines.append("")

    if report["issues"]["exact_duplicate_groups"]:
        lines.append("## Exact Duplicate Groups")
        lines.append("")
        for group in report["issues"]["exact_duplicate_groups"][:20]:
            lines.append(f"- Hash `{group['hash'][:16]}...`: {group['count']} copies")
            for p in group["paths"]:
                lines.append(f"  - `{p}`")
        if len(report["issues"]["exact_duplicate_groups"]) > 20:
            lines.append(
                f"- ... and {len(report['issues']['exact_duplicate_groups']) - 20} more groups"
            )
        lines.append("")

    if report["issues"]["near_duplicate_groups"]:
        lines.append("## Near-Duplicate Groups")
        lines.append("")
        for group in report["issues"]["near_duplicate_groups"][:20]:
            lines.append(
                f"- Distance `{group['hamming_distance']}`: `{group['path_a']}` ↔ `{group['path_b']}`"
            )
        if len(report["issues"]["near_duplicate_groups"]) > 20:
            lines.append(
                f"- ... and {len(report['issues']['near_duplicate_groups']) - 20} more pairs"
            )
        lines.append("")

    if report["issues"]["class_label_conflicts"]:
        lines.append("## Class-Label Conflicts")
        lines.append("")
        for conflict in report["issues"]["class_label_conflicts"]:
            lines.append(
                f"- `{conflict['class_a']}` ↔ `{conflict['class_b']}` (similarity: {conflict['similarity']:.2f})"
            )
        lines.append("")

    if report["issues"]["split_leakage"]:
        lines.append("## Split Leakage")
        lines.append("")
        for leak in report["issues"]["split_leakage"]:
            lines.append(
                f"- `{leak['split_a']}` ↔ `{leak['split_b']}`: {leak['overlapping_hashes']} overlapping hashes"
            )
        lines.append("")

    if report.get("manifest"):
        lines.append("## Manifest Information")
        lines.append("")
        manifest = report["manifest"]
        lines.append(f"- Path: `{manifest.get('path', 'N/A')}`")
        lines.append(
            f"- Schema version: `{manifest.get('schema_version', 'N/A')}`"
        )
        lines.append(f"- Seed: `{manifest.get('seed', 'N/A')}`")
        lines.append(f"- Ratios: `{manifest.get('ratios', {})}`")
        lines.append(
            f"- Skipped datasets: `{manifest.get('skipped_optional_datasets', [])}`"
        )
        lines.append(
            f"- Missing source files: `{manifest.get('missing_source_files', 'N/A')}`"
        )
        lines.append("")

    lines.append("## Raw Dataset")
    lines.append("")
    lines.append(f"- Total raw images: `{report['raw']['total_images']}`")
    lines.append(f"- Raw decode failures: `{report['raw']['decode_failures']}`")
    lines.append(f"- Raw zero-byte files: `{report['raw']['zero_byte_files']}`")
    lines.append("")
    lines.append("| Class | Count |")
    lines.append("| --- | ---: |")
    for cls, count in sorted(
        report["raw"].get("class_distribution", {}).items(),
        key=lambda x: (-x[1], x[0]),
    ):
        lines.append(f"| {cls} | {count} |")
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("- JSON report: `dataset_audit_<timestamp>.json`")
    lines.append("- Markdown report: `dataset_audit_<timestamp>.md`")
    lines.append("")

    return "\n".join(lines) + "\n"


def _unique_output_paths(output_dir: Path, timestamp: str) -> tuple[Path, Path]:
    json_path = output_dir / f"dataset_audit_{timestamp}.json"
    markdown_path = output_dir / f"dataset_audit_{timestamp}.md"
    suffix = 1
    while json_path.exists() or markdown_path.exists():
        json_path = output_dir / f"dataset_audit_{timestamp}_{suffix}.json"
        markdown_path = output_dir / f"dataset_audit_{timestamp}_{suffix}.md"
        suffix += 1
    return json_path, markdown_path


def write_reports(report: dict[str, Any], output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path, markdown_path = _unique_output_paths(output_path, stamp)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the Leaflight dataset quality without modifying data."
    )
    parser.add_argument(
        "--data-root", default="data", help="Root data directory."
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Processed split directory.",
    )
    parser.add_argument(
        "--raw-dir", default="data/raw", help="Raw dataset directory."
    )
    parser.add_argument(
        "--split-manifest",
        default="data/splits/phase1_split.json",
        help="Split manifest path.",
    )
    parser.add_argument(
        "--class-mapping",
        default="data/class_mapping.json",
        help="Class mapping JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON and Markdown reports.",
    )
    args = parser.parse_args(argv)

    report = audit_dataset(
        data_root=args.data_root,
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        split_manifest=args.split_manifest,
        class_mapping_path=args.class_mapping,
    )
    json_path, markdown_path = write_reports(report, args.output_dir)
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
