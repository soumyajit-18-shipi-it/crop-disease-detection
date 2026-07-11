from pathlib import Path

from src.data.dataset import CropDiseaseDataset


def test_dataset_handles_missing_root(tmp_path: Path):
    mapping = tmp_path / "mapping.json"
    mapping.write_text('{"class_to_idx": {"A": 0}, "idx_to_class": {"0": "A"}}', encoding="utf-8")
    dataset = CropDiseaseDataset(tmp_path / "missing", mapping_path=mapping)
    assert len(dataset) == 0
