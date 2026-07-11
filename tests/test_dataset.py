from pathlib import Path

from src.data.dataset import LeafDiseaseDataset


def test_dataset_handles_missing_root(tmp_path: Path):
    dataset = LeafDiseaseDataset(tmp_path / "missing")
    assert len(dataset) == 0
    assert dataset.classes == []
