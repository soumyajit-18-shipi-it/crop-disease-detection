from pathlib import Path

from PIL import Image
import torch

from src.training.config import DatasetSourceConfig, load_config
from src.training.train import _cpu_rng_tensor, train_architecture


class _TinyClassifier(torch.nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Linear(4, classes)

    def forward(self, images):
        return self.classifier(self.features(images).flatten(1))


def test_rng_state_is_normalized_after_cuda_checkpoint_mapping():
    state = torch.get_rng_state()
    mapped = state.to("cuda") if torch.cuda.is_available() else state

    normalized = _cpu_rng_tensor(mapped)

    assert normalized.device.type == "cpu"
    assert normalized.dtype == torch.uint8
    generator = torch.Generator()
    generator.set_state(normalized)


def _make_dataset(root: Path) -> None:
    for split in ("train", "val", "test"):
        for class_index, class_name in enumerate(("healthy", "diseased")):
            directory = root / split / class_name
            directory.mkdir(parents=True)
            for index in range(2):
                value = 40 + class_index * 150 + index
                Image.new("RGB", (40, 36), color=(value, value, value)).save(directory / f"{index}.png")


def test_one_epoch_produces_complete_resumable_and_deployable_artifacts(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    _make_dataset(data_root)
    config = load_config("configs/training/phase2_5.yaml")
    config.experiment_name = "smoke"
    config.data.sources = [
        DatasetSourceConfig("smoke", "pre_split_image_folder", str(data_root))
    ]
    config.data.split_manifest = str(tmp_path / "split.json")
    config.data.image_size = 32
    config.data.use_model_preprocessing = False
    config.data.batch_size = 2
    config.data.num_workers = 0
    config.augmentation.enabled = False
    config.optimization.epochs = 1
    config.optimization.warmup_epochs = 0
    config.optimization.class_weighting = False
    config.runtime.device = "cpu"
    config.runtime.mixed_precision = False
    config.runtime.ema_enabled = True
    config.evaluation.temperature_max_iterations = 5
    config.evaluation.latency_warmup_iterations = 0
    config.evaluation.latency_iterations = 1
    config.output.run_root = str(tmp_path / "runs")

    monkeypatch.setattr(
        "src.training.train.build_model",
        lambda _architecture, classes, *_args, **_kwargs: _TinyClassifier(classes),
    )
    monkeypatch.setattr(
        "src.training.train.resolve_preprocessing",
        lambda _model, image_size, _defaults: {
            "image_size": image_size,
            "resize_mode": "stretch",
            "crop_pct": 1.0,
            "interpolation": "linear",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "input_color_space": "RGB",
            "input_range": [0.0, 1.0],
            "source": "test",
        },
    )

    run_dir = train_architecture(config, "efficientnetv2_s", resume=True)
    expected = {
        "best.pt",
        "last.pt",
        "model.onnx",
        "model.json",
        "metrics.json",
        "evaluation_report.md",
        "evaluation_logits.npz",
        "per_class_metrics.csv",
        "confusion_matrix.json",
        "misclassified_images.json",
        "confidence_distribution.json",
        "checksum.sha256",
        "training_history.csv",
        "confusion_matrix.png",
        "classification_report.json",
        "calibration.json",
        "reliability_diagram.png",
        "run_state.json",
    }
    assert expected <= {path.name for path in run_dir.iterdir()}
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    assert best["checkpoint_type"] == "inference"
    assert "optimizer_state_dict" not in best
    assert last["checkpoint_type"] == "resume"
    assert "optimizer_state_dict" in last
