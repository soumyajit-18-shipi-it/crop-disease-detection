from __future__ import annotations

import json
from pathlib import Path

import pytest

PROVENANCE_PATH = Path("data/provenance/datasets.json")


def _load_provenance() -> dict:
    if not PROVENANCE_PATH.exists():
        pytest.fail("data/provenance/datasets.json does not exist")
    return json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))


def test_provenance_schema_version_present():
    payload = _load_provenance()
    assert "schema_version" in payload
    assert payload["schema_version"] == "1.0"


def test_provenance_governance_policy_present():
    payload = _load_provenance()
    assert "governance_policy" in payload
    policy = payload["governance_policy"]
    assert "version" in policy
    assert "training_eligibility_rule" in policy
    assert "personal_data_rule" in policy
    assert "unknown_fields_policy" in policy
    assert "fabrication_policy" in policy


def test_provenance_datasets_is_list():
    payload = _load_provenance()
    assert isinstance(payload.get("datasets"), list)
    assert len(payload["datasets"]) >= 1


def test_provenance_required_fields():
    payload = _load_provenance()
    required_top = {"schema_version", "generated_at", "datasets", "governance_policy"}
    assert required_top.issubset(payload.keys())


def test_provenance_each_dataset_has_required_fields():
    payload = _load_provenance()
    required = {
        "name",
        "display_name",
        "source",
        "version",
        "checksum",
        "license",
        "allowed_usage",
        "crop_class_coverage",
        "review_status",
        "personal_data_risk",
        "split_eligibility",
        "storage",
        "maintainer_notes",
    }
    for dataset in payload["datasets"]:
        assert required.issubset(dataset.keys()), f"Dataset {dataset.get('name')} missing fields"


def test_provenance_plantvillage_approved():
    payload = _load_provenance()
    plantvillage = next((d for d in payload["datasets"] if d["name"] == "plantvillage"), None)
    assert plantvillage is not None
    assert plantvillage["review_status"] == "approved_for_training"
    assert plantvillage["split_eligibility"]["eligible"] is True
    assert plantvillage["personal_data_risk"] == "none"


def test_provenance_plantdoc_pending():
    payload = _load_provenance()
    plantdoc = next((d for d in payload["datasets"] if d["name"] == "plantdoc"), None)
    assert plantdoc is not None
    assert plantdoc["review_status"] == "pending_ingestion"
    assert plantdoc["split_eligibility"]["eligible"] is False


def test_provenance_field_survey_pending_review():
    payload = _load_provenance()
    field = next((d for d in payload["datasets"] if d["name"] == "field_survey"), None)
    assert field is not None
    assert field["review_status"] == "pending_manual_approval"
    assert field["split_eligibility"]["eligible"] is False
    assert field["personal_data_risk"]["level"] == "high"
    assert "Name_of_the_Surveyer" in field["personal_data_risk"]["fields"]


def test_provenance_no_fabricated_licences():
    payload = _load_provenance()
    for dataset in payload["datasets"]:
        license_info = dataset.get("license", {})
        name = license_info.get("name")
        spdx = license_info.get("spdx_id")
        if name and name not in {"unknown", "Not verified"}:
            pytest.fail(f"Dataset {dataset['name']} has potentially fabricated licence name: {name}")
        if spdx and spdx not in {"unknown", "Not verified"}:
            pytest.fail(f"Dataset {dataset['name']} has potentially fabricated licence SPDX: {spdx}")


def test_provenance_no_fabricated_checksums():
    payload = _load_provenance()
    for dataset in payload["datasets"]:
        checksum = dataset.get("checksum", {})
        expected = checksum.get("expected_sha256")
        if expected and expected != "unknown":
            pytest.fail(f"Dataset {dataset['name']} has a non-unknown checksum without verification")


def test_provenance_source_references_are_paths_or_slugs():
    payload = _load_provenance()
    for dataset in payload["datasets"]:
        source = dataset.get("source", {})
        slug = source.get("slug")
        reference = source.get("reference")
        if slug and slug not in {"unknown"}:
            assert "/" in slug or source.get("type") == "kaggle"
        if reference and reference not in {"unknown", None}:
            path = Path(reference)
            assert path.exists() or reference.startswith("src/") or reference.startswith("data/")


def test_provenance_storage_paths_are_strings():
    payload = _load_provenance()
    for dataset in payload["datasets"]:
        storage = dataset.get("storage", {})
        for key in ("raw_path", "processed_path", "split_manifest", "manifest_path",
                    "cleaned_manifest_path", "validated_manifest_path", "decisions_log"):
            value = storage.get(key)
            if value is not None:
                assert isinstance(value, str), f"Dataset {dataset['name']} storage.{key} must be string or null"


def test_provenance_allowed_usage_is_list():
    payload = _load_provenance()
    for dataset in payload["datasets"]:
        assert isinstance(dataset.get("allowed_usage"), list)


def test_provenance_split_eligibility_has_required_fields():
    payload = _load_provenance()
    for dataset in payload["datasets"]:
        eligibility = dataset.get("split_eligibility", {})
        assert "eligible" in eligibility
        assert isinstance(eligibility["eligible"], bool)
