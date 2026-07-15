from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from src.data.dataset import get_dataloaders
from src.data.multisource_dataset import create_dataloaders
from src.evaluation.calibration import softmax_probabilities
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix
from src.models.baseline_cnn import BaselineCNN
from src.models.model_factory import build_model
from src.training.config import config_from_dict, detect_device
from src.training.engine import run_epoch


def _load_checkpoint(path: str | Path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata = checkpoint["metadata"]
    architecture = metadata["architecture"]
    num_classes = metadata["num_classes"]
    model = BaselineCNN(num_classes) if architecture == "baseline_cnn" else build_model(architecture, num_classes, False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, metadata


def evaluate(checkpoint_path: str = "models/checkpoints/best_model.pth") -> Path:
    device = detect_device()
    model, metadata = _load_checkpoint(checkpoint_path, device)
    config_payload = metadata["config"]
    if "data" in config_payload:
        config = config_from_dict(config_payload)
        config.runtime.device = device
        manifest_path = Path(metadata.get("split_manifest", config.data.split_manifest))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        test_loader = create_dataloaders(
            config,
            manifest=manifest,
            preprocessing=metadata.get("preprocessing"),
        )[0]["test"]
    else:
        _, _, test_loader = get_dataloaders(
            batch_size=config_payload["batch_size"],
            num_workers=config_payload["num_workers"],
            data_dir=config_payload["data_dir"],
            mapping_path=config_payload["mapping_path"],
            image_size=config_payload["image_size"],
        )
    criterion = torch.nn.CrossEntropyLoss()
    result = run_epoch(model, test_loader, criterion, device)
    test_loss = result["loss"]
    test_acc = result["accuracy"]
    y_pred = result["predictions"]
    y_true = result["targets"]
    class_names = [metadata["idx_to_class"][str(i)] for i in range(metadata["num_classes"])]
    probabilities = softmax_probabilities(
        result["logits"],
        float(metadata.get("calibration", {}).get("temperature", 1.0)),
    )
    metrics = compute_metrics(y_true, y_pred, class_names, probabilities)
    matrix_path = plot_confusion_matrix(y_true, y_pred, class_names)

    confused = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    top_pairs = confused.most_common(10)
    report_path = Path("docs/model_performance_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        file.write("# Model Performance Report\n\n")
        file.write(f"Checkpoint: `{checkpoint_path}`\n\n")
        file.write(f"Test loss: {test_loss:.4f}\n\n")
        file.write(f"Test accuracy: {test_acc:.4f}\n\n")
        file.write(f"Macro F1: {metrics['macro']['f1']:.4f}\n\n")
        file.write(f"Confusion matrix: `{matrix_path}`\n\n")
        file.write("## Per-Class Metrics\n\n")
        file.write("| Class | Precision | Recall | F1 | Support |\n|---|---:|---:|---:|---:|\n")
        for class_name, row in metrics["per_class"].items():
            file.write(
                f"| {class_name} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1']:.4f} | {row['support']} |\n"
            )
        file.write("\n## Top Confused Pairs\n\n")
        if not top_pairs:
            file.write("No misclassifications found in this evaluation run.\n")
        for (true_idx, pred_idx), count in top_pairs:
            file.write(f"- {class_names[true_idx]} -> {class_names[pred_idx]}: {count}\n")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_model.pth")
    args = parser.parse_args()
    print(f"Wrote report to {evaluate(args.checkpoint)}")


if __name__ == "__main__":
    main()
