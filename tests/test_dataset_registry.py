import json

import pytest
from PIL import Image

from src.data.multisource_dataset import build_split_manifest
from src.data.registry import load_registered_sources
from src.training.config import DataConfig, DatasetSourceConfig, TrainConfig


def test_registry_refuses_detailed_field_survey_review_manifest(tmp_path):
    manifest = tmp_path / "validated_manifest.json"
    manifest.write_text(json.dumps({"records": [
        {"record_id": "yes", "image_path": "leaf.jpg", "validation": {"eligible_for_training": True, "canonical_disease": "Blight"}},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="not training-safe|Refusing"):
        load_registered_sources([DatasetSourceConfig("field", "validated_manifest", str(manifest), optional=True)])


def test_registry_uses_sanitized_field_survey_training_manifest_only(tmp_path, monkeypatch):
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = image_root / "leaf.jpg"
    Image.new("RGB", (8, 8)).save(image)
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "1.0",
        "manifest_type": "field_survey_training",
        "dataset_name": "field_survey",
        "dataset_source": "field_survey",
        "review_manifest_sha256": "a" * 64,
        "statistics": {
            "total_source_records": 1,
            "accepted_records": 1,
            "replaced_records": 0,
            "rejected_records": 0,
            "pending_records": 0,
            "exported_records": 1,
            "excluded_invalid_records": 0,
        },
        "class_names": ["Blight"],
        "records": [
            {
                "record_id": "yes",
                "dataset_source": "field_survey",
                "image_path": "images/leaf.jpg",
                "image_sha256": "b" * 64,
                "canonical_class": "Blight",
                "review_status": "validated",
                "review_decision": "accept",
            },
        ],
    }), encoding="utf-8")
    (tmp_path / "validated_manifest.json").write_text(json.dumps({"records": [
        {"record_id": "review-only", "image_path": str(image), "validation": {"eligible_for_training": True, "canonical_disease": "Other"}},
    ]}), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    records, _ = load_registered_sources([
        DatasetSourceConfig("field", "field_survey_training_manifest", str(manifest))
    ])
    assert len(records) == 1
    assert records[0].label == "Blight"


def test_split_manifest_is_reproducible(tmp_path):
    root = tmp_path / "images" / "Disease"
    root.mkdir(parents=True)
    for index in range(10):
        Image.new("RGB", (8, 8), (index, 0, 0)).save(root / f"{index}.png")
    split_path = tmp_path / "split.json"
    config = TrainConfig(data=DataConfig(
        sources=[DatasetSourceConfig("source", "image_folder", str(tmp_path / "images"))],
        split_manifest=str(split_path), num_workers=0,
    ))
    first = build_split_manifest(config)
    second = build_split_manifest(config)
    assert first == second
    assert {item["split"] for item in first["items"]} == {"train", "val", "test"}
