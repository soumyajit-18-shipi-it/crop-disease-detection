from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.inference.preprocess_input import preprocess_for_model, preprocess_for_onnx
from src.models.baseline_cnn import BaselineCNN
from src.models.model_factory import build_model


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def load_model(checkpoint_path: str | Path, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = checkpoint["metadata"]
    architecture = metadata["architecture"]
    num_classes = metadata["num_classes"]
    model = BaselineCNN(num_classes) if architecture == "baseline_cnn" else build_model(architecture, num_classes, False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, metadata


@torch.no_grad()
def predict(model, image, metadata: dict, device: str | None = None) -> dict:
    device = device or next(model.parameters()).device
    image_size = int(metadata.get("image_size", 224))
    tensor = preprocess_for_model(image, image_size, metadata.get("preprocessing")).to(device)
    temperature = float(metadata.get("calibration", {}).get("temperature", 1.0))
    logits = model(tensor) / temperature
    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx_to_class = {int(k): v for k, v in metadata["idx_to_class"].items()}
    top_indices = probabilities.argsort()[-3:][::-1]
    top_3 = [{"class_name": idx_to_class[int(i)], "confidence": float(probabilities[i])} for i in top_indices]
    return {
        "class_name": top_3[0]["class_name"],
        "confidence": top_3[0]["confidence"],
        "top_3_predictions": top_3,
    }


def export_to_onnx(checkpoint_path: str | Path, output_path: str | Path = "models/onnx/model.onnx") -> Path:
    model, metadata = load_model(checkpoint_path, "cpu")
    image_size = int(metadata.get("image_size", 224))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    json_metadata = {
        "architecture": metadata["architecture"],
        "num_classes": metadata["num_classes"],
        "image_size": metadata.get("image_size", 224),
        "idx_to_class": metadata["idx_to_class"],
        "preprocessing": metadata.get("preprocessing"),
        "calibration": metadata.get("calibration", {"temperature": 1.0}),
    }
    with output.with_suffix(".json").open("w", encoding="utf-8") as file:
        json.dump(json_metadata, file, indent=2)
    return output


def verify_onnx(checkpoint_path: str | Path, onnx_path: str | Path, image) -> bool:
    import onnxruntime as ort

    model, metadata = load_model(checkpoint_path, "cpu")
    model.eval()
    preprocessing = metadata.get("preprocessing")
    torch_input = preprocess_for_model(image, int(metadata.get("image_size", 224)), preprocessing)
    onnx_input = preprocess_for_onnx(image, int(metadata.get("image_size", 224)), preprocessing)
    with torch.no_grad():
        torch_logits = model(torch_input).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"input": onnx_input})[0]
    return bool(np.allclose(torch_logits, onnx_logits, rtol=1e-3, atol=1e-4))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_model.pth")
    parser.add_argument("--output", default="models/onnx/model.onnx")
    args = parser.parse_args()
    print(f"Exported ONNX model to {export_to_onnx(args.checkpoint, args.output)}")


if __name__ == "__main__":
    main()
