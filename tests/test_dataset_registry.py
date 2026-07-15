import json

from PIL import Image

from src.data.multisource_dataset import build_split_manifest
from src.data.registry import load_registered_sources
from src.training.config import DataConfig, DatasetSourceConfig, TrainConfig


def test_validated_manifest_hard_gates_field_survey(tmp_path):
    image = tmp_path / "leaf.jpg"
    Image.new("RGB", (8, 8)).save(image)
    manifest = tmp_path / "validated.json"
    manifest.write_text(json.dumps({"records": [
        {"record_id": "no", "image_path": str(image), "validation": {"eligible_for_training": False, "canonical_disease": "Blight"}},
        {"record_id": "yes", "image_path": str(image), "validation": {"eligible_for_training": True, "canonical_disease": "Blight"}},
    ]}), encoding="utf-8")
    records, _ = load_registered_sources([DatasetSourceConfig("field", "validated_manifest", str(manifest))])
    assert len(records) == 1


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
