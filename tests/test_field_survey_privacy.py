from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.data.validate_field_survey_manifest import (
    PrivacyAuditError,
    audit_privacy,
    validate_privacy_policy,
)


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_audit_privacy_detects_personal_metadata(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, {
        "schema_version": "1.0",
        "records": [
            {
                "record_id": "1",
                "image_path": "images/a.jpg",
                "metadata": {
                    "_id": "123",
                    "_uuid": "abc",
                    "_submission_time": "2023-01-01",
                    "Name_of_the_Surveyer": "John",
                    "_Farmer_Name": "Jane",
                    "start": "2023-01-01T00:00:00",
                    "end": "2023-01-01T01:00:00",
                    "Team_Number": "1",
                    "College_name": "ABC",
                },
                "validation": {"eligible_for_training": False, "status": "pending"},
            }
        ],
        "source_issues": {
            "duplicate_image_hashes": [],
            "duplicate_image_reference_groups": [],
        },
        "statistics": {
            "missing_images": 0,
            "invalid_labels": 0,
        },
    })
    report = audit_privacy(manifest)
    assert report["overall_status"] == "fail"
    assert any(f["check"] == "no_personal_metadata_in_training_manifests" for f in report["failures"])
    assert report["summary"]["records_with_personal_metadata"] == 1
    assert report["summary"]["absolute_image_paths"] == 0


def test_audit_privacy_detects_absolute_paths(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    absolute_path = "C:/absolute/path/a.jpg" if sys.platform == "win32" else "/absolute/path/a.jpg"
    _write_manifest(manifest, {
        "schema_version": "1.0",
        "records": [
            {
                "record_id": "1",
                "image_path": absolute_path,
                "metadata": {},
                "validation": {"eligible_for_training": False, "status": "pending"},
            }
        ],
        "source_issues": {
            "duplicate_image_hashes": [],
            "duplicate_image_reference_groups": [],
        },
        "statistics": {
            "missing_images": 0,
            "invalid_labels": 0,
        },
    })
    report = audit_privacy(manifest)
    assert any(f["check"] == "portable_image_paths" for f in report["failures"])
    assert report["summary"]["absolute_image_paths"] == 1


def test_audit_privacy_detects_eligible_records(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, {
        "schema_version": "1.0",
        "records": [
            {
                "record_id": "1",
                "image_path": "images/a.jpg",
                "metadata": {},
                "validation": {"eligible_for_training": True, "status": "accepted"},
            }
        ],
        "source_issues": {
            "duplicate_image_hashes": [],
            "duplicate_image_reference_groups": [],
        },
        "statistics": {
            "missing_images": 0,
            "invalid_labels": 0,
        },
    })
    report = audit_privacy(manifest)
    assert any(f["check"] == "no_eligible_records_without_review" for f in report["failures"])
    assert report["summary"]["eligible_for_training"] == 1


def test_audit_privacy_passes_when_clean(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, {
        "schema_version": "1.0",
        "records": [
            {
                "record_id": "1",
                "image_path": "images/a.jpg",
                "metadata": {"Name_of_the_Surveyer": ""},
                "validation": {"eligible_for_training": False, "status": "pending"},
            }
        ],
        "source_issues": {
            "duplicate_image_hashes": [],
            "duplicate_image_reference_groups": [],
        },
        "statistics": {
            "missing_images": 0,
            "invalid_labels": 0,
        },
    })
    report = audit_privacy(manifest)
    assert report["overall_status"] == "pass"
    assert not any(f["check"] == "no_personal_metadata_in_training_manifests" for f in report["failures"])
    assert not any(f["check"] == "no_eligible_records_without_review" for f in report["failures"])


def test_audit_privacy_detects_missing_images(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, {
        "schema_version": "1.0",
        "records": [
            {
                "record_id": "1",
                "image_path": "missing.jpg",
                "metadata": {},
                "validation": {"eligible_for_training": False, "status": "pending"},
            }
        ],
        "source_issues": {
            "duplicate_image_hashes": [],
            "duplicate_image_reference_groups": [],
        },
        "statistics": {
            "missing_images": 1,
            "invalid_labels": 0,
        },
    })
    report = audit_privacy(manifest)
    assert any(f["check"] == "no_missing_images" for f in report["failures"])
    assert report["summary"]["missing_images"] == 1


def test_audit_privacy_detects_duplicate_groups(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, {
        "schema_version": "1.0",
        "records": [
            {
                "record_id": "1",
                "image_path": "a.jpg",
                "metadata": {},
                "validation": {"eligible_for_training": False, "status": "pending"},
            }
        ],
        "source_issues": {
            "duplicate_image_hashes": [{"sha256": "abc", "record_ids": ["1", "2"]}],
            "duplicate_image_reference_groups": [],
        },
        "statistics": {
            "missing_images": 0,
            "invalid_labels": 0,
        },
    })
    report = audit_privacy(manifest)
    assert any(f["check"] == "no_duplicate_groups" for f in report["failures"])
    assert report["summary"]["duplicate_hash_groups"] == 1


def test_validate_privacy_policy_raises_on_hard_failure(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, {
        "schema_version": "1.0",
        "records": [
            {
                "record_id": "1",
                "image_path": "images/a.jpg",
                "metadata": {"_Farmer_Name": "Jane"},
                "validation": {"eligible_for_training": True, "status": "accepted"},
            }
        ],
        "source_issues": {
            "duplicate_image_hashes": [],
            "duplicate_image_reference_groups": [],
        },
        "statistics": {
            "missing_images": 0,
            "invalid_labels": 0,
        },
    })
    with pytest.raises(PrivacyAuditError):
        validate_privacy_policy(manifest)


def test_validate_privacy_policy_passes_when_soft_failures_only(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, {
        "schema_version": "1.0",
        "records": [
            {
                "record_id": "1",
                "image_path": "images/a.jpg",
                "metadata": {},
                "validation": {"eligible_for_training": False, "status": "pending"},
            }
        ],
        "source_issues": {
            "duplicate_image_hashes": [],
            "duplicate_image_reference_groups": [],
        },
        "statistics": {
            "missing_images": 2,
            "invalid_labels": 0,
        },
    })
    report = validate_privacy_policy(manifest)
    assert report["overall_status"] == "fail"
    assert any(f["check"] == "no_missing_images" for f in report["failures"])


def test_audit_privacy_counts_review_progress(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, {
        "schema_version": "1.0",
        "records": [
            {
                "record_id": "1",
                "image_path": "images/a.jpg",
                "metadata": {},
                "validation": {"eligible_for_training": False, "status": "pending"},
            }
        ],
        "audit_history": [
            {"action": "accept"},
            {"action": "replace"},
            {"action": "reject"},
        ],
        "source_issues": {
            "duplicate_image_hashes": [],
            "duplicate_image_reference_groups": [],
        },
        "statistics": {
            "missing_images": 0,
            "invalid_labels": 0,
        },
    })
    report = audit_privacy(manifest)
    assert report["review_progress"]["accepted"] == 1
    assert report["review_progress"]["replaced"] == 1
    assert report["review_progress"]["rejected"] == 1


def test_audit_privacy_handles_missing_manifest(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        audit_privacy(tmp_path / "nonexistent.json")


def test_audit_privacy_personal_field_examples_are_truncated(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    records = [
        {
            "record_id": str(i),
            "image_path": "images/a.jpg",
            "metadata": {"_Farmer_Name": f"Farmer{i}"},
            "validation": {"eligible_for_training": False, "status": "pending"},
        }
        for i in range(5)
    ]
    _write_manifest(manifest, {
        "schema_version": "1.0",
        "records": records,
        "source_issues": {
            "duplicate_image_hashes": [],
            "duplicate_image_reference_groups": [],
        },
        "statistics": {
            "missing_images": 0,
            "invalid_labels": 0,
        },
    })
    report = audit_privacy(manifest)
    examples = report["personal_fields"]["examples"]["_Farmer_Name"]
    assert len(examples) == 3
    assert examples[0] == "Farmer0"


def test_audit_privacy_null_image_paths_counted(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, {
        "schema_version": "1.0",
        "records": [
            {
                "record_id": "1",
                "image_path": None,
                "metadata": {},
                "validation": {"eligible_for_training": False, "status": "pending"},
            },
            {
                "record_id": "2",
                "image_path": "",
                "metadata": {},
                "validation": {"eligible_for_training": False, "status": "pending"},
            },
        ],
        "source_issues": {
            "duplicate_image_hashes": [],
            "duplicate_image_reference_groups": [],
        },
        "statistics": {
            "missing_images": 0,
            "invalid_labels": 0,
        },
    })
    report = audit_privacy(manifest)
    assert report["summary"]["null_image_paths"] == 2
