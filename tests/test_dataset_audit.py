from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.audit_dataset import (
    _analyze_image,
    _class_imbalance_ratios,
    _class_label_conflicts,
    _collect_raw_images,
    _collect_split_images,
    _detect_split_leakage,
    _find_exact_duplicates,
    _find_near_duplicates,
    _read_split_manifest,
    audit_dataset,
    render_markdown,
    write_reports,
)


def _write_image(path: Path, color: tuple[int, int, int] = (128, 128, 128), size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def test_analyze_image_valid(tmp_path: Path):
    image = tmp_path / "leaf.jpg"
    _write_image(image)
    record = _analyze_image(image)
    assert record.width == 32
    assert record.height == 32
    assert record.color_mode == "RGB"
    assert record.size_bytes > 0
    assert record.exact_hash is not None
    assert record.perceptual_hash is not None
    assert record.decode_error is None


def test_analyze_image_decode_failure(tmp_path: Path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    record = _analyze_image(bad)
    assert record.decode_error is not None
    assert record.exact_hash is None
    assert record.perceptual_hash is None


def test_analyze_image_zero_byte(tmp_path: Path):
    zero = tmp_path / "zero.jpg"
    zero.write_bytes(b"")
    record = _analyze_image(zero)
    assert record.size_bytes == 0
    assert record.decode_error is not None


def test_find_exact_duplicates_empty():
    assert _find_exact_duplicates([]) == []


def test_find_exact_duplicates_with_groups(tmp_path: Path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    c = tmp_path / "c.jpg"
    _write_image(a, color=(10, 20, 30))
    _write_image(b, color=(10, 20, 30))
    _write_image(c, color=(40, 50, 60))
    records = [_analyze_image(a), _analyze_image(b), _analyze_image(c)]
    groups = _find_exact_duplicates(records)
    assert len(groups) == 1
    assert groups[0]["count"] == 2
    assert len(groups[0]["paths"]) == 2


def test_find_near_duplicates_empty():
    assert _find_near_duplicates([]) == []


def test_find_near_duplicates_with_similar(tmp_path: Path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    _write_image(a, color=(100, 100, 100))
    _write_image(b, color=(105, 100, 100))
    records = [_analyze_image(a), _analyze_image(b)]
    pairs = _find_near_duplicates(records, hamming_threshold=15)
    assert len(pairs) >= 1
    assert pairs[0]["hamming_distance"] <= 15


def test_class_label_conflicts_detects_similar():
    conflicts = _class_label_conflicts(["Tomato_healthy", "Tomato_Leaf_Mold", "Tomato_healthy_copy"])
    assert len(conflicts) >= 1
    assert any(c["class_a"] == "Tomato_healthy" and c["class_b"] == "Tomato_healthy_copy" for c in conflicts)


def test_class_label_conflicts_no_conflicts():
    assert _class_label_conflicts(["Apple", "Banana", "Cherry"], threshold=0.9) == []


def test_class_imbalance_ratios_empty():
    assert _class_imbalance_ratios({}) == {
        "max_class_share": 0.0,
        "min_class_share": 0.0,
        "imbalance_ratio": 0.0,
    }


def test_class_imbalance_ratios_computed():
    ratios = _class_imbalance_ratios({"A": 100, "B": 10, "C": 50})
    assert ratios["max_class_share"] == 100 / 160
    assert ratios["min_class_share"] == 10 / 160
    assert ratios["imbalance_ratio"] == 10.0


def test_detect_split_leakage_none(tmp_path: Path):
    train = tmp_path / "train" / "A"
    val = tmp_path / "val" / "A"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    a1 = train / "1.jpg"
    a2 = val / "2.jpg"
    _write_image(a1, color=(10, 10, 10))
    _write_image(a2, color=(20, 20, 20))
    records = {
        "train": [_analyze_image(a1)],
        "val": [_analyze_image(a2)],
        "test": [],
    }
    assert _detect_split_leakage(records) == []


def test_detect_split_leakage_found(tmp_path: Path):
    train = tmp_path / "train" / "A"
    val = tmp_path / "val" / "A"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    a1 = train / "1.jpg"
    a2 = val / "1.jpg"
    _write_image(a1, color=(30, 30, 30))
    _write_image(a2, color=(30, 30, 30))
    records = {
        "train": [_analyze_image(a1)],
        "val": [_analyze_image(a2)],
        "test": [],
    }
    leakage = _detect_split_leakage(records)
    assert len(leakage) == 1
    assert leakage[0]["split_a"] == "train"
    assert leakage[0]["split_b"] == "val"
    assert leakage[0]["overlapping_hashes"] == 1


def test_read_split_manifest_missing(tmp_path: Path):
    assert _read_split_manifest(tmp_path / "nonexistent.json") is None


def test_read_split_manifest_valid(tmp_path: Path):
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"schema_version": "1.0", "seed": 42}), encoding="utf-8")
    payload = _read_split_manifest(manifest)
    assert payload is not None
    assert payload["seed"] == 42


def test_audit_dataset_minimal(tmp_path: Path):
    processed = tmp_path / "processed"
    train = processed / "train" / "Tomato_healthy"
    val = processed / "val" / "Tomato_healthy"
    test = processed / "test" / "Tomato_healthy"
    for d in (train, val, test):
        d.mkdir(parents=True)
    _write_image(train / "a.jpg", color=(10, 20, 30))
    _write_image(val / "b.jpg", color=(40, 50, 60))
    _write_image(test / "c.jpg", color=(70, 80, 90))

    mapping = tmp_path / "class_mapping.json"
    mapping.write_text(
        json.dumps({"class_to_idx": {"Tomato_healthy": 0}, "idx_to_class": {"0": "Tomato_healthy"}}),
        encoding="utf-8",
    )

    report = audit_dataset(
        data_root=tmp_path,
        processed_dir=processed,
        raw_dir=tmp_path / "raw",
        split_manifest=None,
        class_mapping_path=mapping,
    )
    assert report["class_count"] == 1
    assert report["splits"]["train"] == 1
    assert report["splits"]["val"] == 1
    assert report["splits"]["test"] == 1
    assert report["statistics"]["total_images"] == 3
    assert report["issues"]["exact_duplicate_groups"] == []
    assert report["issues"]["split_leakage"] == []


def test_audit_dataset_detects_issues(tmp_path: Path):
    processed = tmp_path / "processed"
    train = processed / "train" / "Tomato_healthy"
    val = processed / "val" / "Tomato_healthy"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    a1 = train / "a.jpg"
    a2 = val / "a.jpg"
    _write_image(a1, color=(50, 50, 50))
    _write_image(a2, color=(50, 50, 50))

    bad = train / "bad.jpg"
    bad.write_bytes(b"corrupt")

    zero = train / "zero.jpg"
    zero.write_bytes(b"")

    mapping = tmp_path / "class_mapping.json"
    mapping.write_text(
        json.dumps({"class_to_idx": {"Tomato_healthy": 0}, "idx_to_class": {"0": "Tomato_healthy"}}),
        encoding="utf-8",
    )

    report = audit_dataset(
        data_root=tmp_path,
        processed_dir=processed,
        raw_dir=tmp_path / "raw",
        split_manifest=None,
        class_mapping_path=mapping,
    )
    assert len(report["issues"]["exact_duplicate_groups"]) == 1
    assert len(report["issues"]["split_leakage"]) == 1
    assert len(report["issues"]["decode_failures"]) == 2
    assert len(report["issues"]["zero_byte_files"]) == 1


def test_write_reports_creates_files(tmp_path: Path):
    report = {
        "schema_version": "1.0",
        "timestamp": "2026-01-01T00:00:00Z",
        "dataset_root": str(tmp_path),
        "processed_dir": str(tmp_path),
        "raw_dir": str(tmp_path),
        "class_count": 0,
        "classes": [],
        "splits": {},
        "class_distribution": {},
        "raw": {"total_images": 0, "class_distribution": {}, "decode_failures": 0, "zero_byte_files": 0},
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
        "statistics": {
            "total_images": 0,
            "train_ratio": 0.0,
            "val_ratio": 0.0,
            "test_ratio": 0.0,
            "class_imbalance": {},
        },
    }
    output_dir = tmp_path / "reports"
    json_path, md_path = write_reports(report, output_dir)
    assert json_path.exists()
    assert md_path.exists()
    assert json_path.suffix == ".json"
    assert md_path.suffix == ".md"
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "1.0"


def test_render_markdown_contains_sections():
    report = {
        "schema_version": "1.0",
        "timestamp": "2026-01-01T00:00:00Z",
        "dataset_root": "/data",
        "processed_dir": "/data/processed",
        "raw_dir": "/data/raw",
        "class_count": 2,
        "classes": ["A", "B"],
        "splits": {"train": 10, "val": 5, "test": 5},
        "class_distribution": {
            "train": {"A": 6, "B": 4},
            "val": {"A": 3, "B": 2},
            "test": {"A": 2, "B": 3},
        },
        "raw": {"total_images": 0, "class_distribution": {}, "decode_failures": 0, "zero_byte_files": 0},
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
        "statistics": {
            "total_images": 20,
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "class_imbalance": {"max_class_share": 0.6, "min_class_share": 0.4, "imbalance_ratio": 1.5},
        },
    }
    md = render_markdown(report)
    assert "# Dataset Audit Report" in md
    assert "## Split Sizes" in md
    assert "## Class Distribution" in md
    assert "## Class Imbalance" in md
    assert "train" in md
