import json

import torch

from src.training.export_onnx import export_and_verify_onnx


def test_onnx_export_has_metadata_and_parity(tmp_path):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2)).eval()
    output = tmp_path / "model.onnx"
    report = export_and_verify_onnx(
        model, output,
        {"image_size": 8, "idx_to_class": {"0": "healthy", "1": "disease"}},
        image_size=8,
    )
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert output.exists()
    assert report["passed"] is True
    assert metadata["onnx"]["parity"]["passed"] is True
