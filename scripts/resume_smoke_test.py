"""Dry-run validation for a resumable training checkpoint.

The script restores every persisted training state, runs one real training
batch and one real validation batch in memory, and verifies that the source
checkpoints were not modified. It never acquires the production run lock and
never writes training artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
from timm.optim import create_optimizer_v2

from src.data.multisource_dataset import build_split_manifest, create_dataloaders
from src.models.model_factory import build_model
from src.training.config import load_config, validate_config
from src.training.engine import ModelEMA, build_class_weighted_loss, run_epoch
from src.training.train import (
    _apply_architecture_override,
    _cpu_rng_tensor,
    _restore_rng_state,
    _scheduler,
    _training_signature,
    _validate_resume_checkpoint,
)
from src.utils.seed import set_seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _SingleBatch:
    def __init__(self, batch):
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        yield self.batch


def run_smoke_test(config_path: str, architecture: str, checkpoint_path: str | Path) -> dict:
    checkpoint_path = Path(checkpoint_path)
    best_path = checkpoint_path.with_name("best.pt")
    before_hashes = {
        checkpoint_path.name: _sha256(checkpoint_path),
        best_path.name: _sha256(best_path),
    }

    config = load_config(config_path)
    _apply_architecture_override(config, architecture)
    validate_config(config)
    set_seed(config.runtime.seed)
    device = config.resolved_device()
    if not device.startswith("cuda"):
        raise RuntimeError("Resume smoke test requires the original CUDA training environment")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_metadata = state["metadata"]
    manifest = build_split_manifest(config, force=False)
    split_path = Path(config.data.split_manifest)
    split_hash = _sha256(split_path)
    preprocessing = saved_metadata["preprocessing"]
    config.data.image_size = int(preprocessing["image_size"])
    # Worker count does not affect the training signature. Keeping smoke-test
    # loading in-process avoids leaving spawned workers after a one-batch run.
    config.data.num_workers = 0
    loaders, _ = create_dataloaders(
        config, force_split=False, manifest=manifest, preprocessing=preprocessing
    )
    signature = _training_signature(config, architecture, split_hash, preprocessing)
    current_metadata = {
        "architecture": architecture,
        "num_classes": len(manifest["class_to_idx"]),
        "split_manifest_sha256": split_hash,
        "training_signature": signature,
    }
    _validate_resume_checkpoint(state, current_metadata)

    model = build_model(
        architecture,
        current_metadata["num_classes"],
        False,
        config.model.dropout,
        config.model.drop_path_rate,
    ).to(device)
    optimizer = create_optimizer_v2(
        model,
        opt="adamw",
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
        betas=tuple(config.optimization.betas),
        eps=config.optimization.eps,
        filter_bias_and_bn=config.optimization.filter_bias_and_norm_from_weight_decay,
        layer_decay=config.optimization.layer_decay,
    )
    optimizer_steps_per_epoch = math.ceil(
        len(loaders["train"]) / config.optimization.gradient_accumulation_steps
    )
    scheduler = _scheduler(
        optimizer,
        optimizer_steps_per_epoch,
        config.optimization.epochs,
        config.optimization.warmup_epochs,
        config.optimization.warmup_start_factor,
        config.optimization.min_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.runtime.mixed_precision)
    ema = ModelEMA(model, config.runtime.ema_decay)
    criterion = build_class_weighted_loss(
        loaders["train"].dataset,
        current_metadata["num_classes"],
        device,
        config.optimization.label_smoothing,
        config.optimization.class_weight_strategy,
        config.optimization.effective_num_beta,
    )

    model.load_state_dict(state["training_model_state_dict"])
    ema.load_state_dict(state["ema_state_dict"], state["ema_num_updates"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    scaler.load_state_dict(state["scaler_state_dict"])
    _restore_rng_state(state["rng_state"])
    loaders["train"].crop_generator.set_state(_cpu_rng_tensor(state["train_generator_state"]))

    torch.cuda.reset_peak_memory_stats()
    memory_before = int(torch.cuda.memory_allocated())
    train_batch = next(iter(loaders["train"]))
    train_result = run_epoch(
        model,
        _SingleBatch(train_batch),
        criterion,
        device,
        optimizer,
        scaler,
        scheduler,
        ema,
        config.runtime.mixed_precision,
        config.optimization.gradient_clip_norm,
        config.optimization.gradient_accumulation_steps,
        config.augmentation,
        config.runtime.amp_dtype,
    )
    validation_batch = next(iter(loaders["val"]))
    validation_result = run_epoch(ema.module, _SingleBatch(validation_batch), criterion, device)
    memory_after = int(torch.cuda.memory_allocated())
    memory_peak = int(torch.cuda.max_memory_allocated())
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)

    after_hashes = {
        checkpoint_path.name: _sha256(checkpoint_path),
        best_path.name: _sha256(best_path),
    }
    report = {
        "passed": bool(
            math.isfinite(train_result["loss"])
            and math.isfinite(validation_result["loss"])
            and math.isfinite(train_result["maximum_gradient_norm"])
            and train_result["optimizer_steps"] == 1
            and memory_peak < total_memory
            and before_hashes == after_hashes
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(state["epoch"]),
        "next_epoch": int(state["epoch"]) + 1,
        "architecture": architecture,
        "split_manifest_sha256": split_hash,
        "training_signature": signature,
        "optimizer_state_entries": len(state["optimizer_state_dict"]["state"]),
        "scheduler_last_epoch": int(state["scheduler_state_dict"]["last_epoch"]),
        "ema_num_updates": int(state["ema_num_updates"]),
        "amp_scale": float(state["scaler_state_dict"]["scale"]),
        "rng_state_restored": True,
        "dataloader_generator_state_restored": True,
        "train_loss": float(train_result["loss"]),
        "validation_loss": float(validation_result["loss"]),
        "maximum_gradient_norm": float(train_result["maximum_gradient_norm"]),
        "optimizer_steps": int(train_result["optimizer_steps"]),
        "skipped_optimizer_steps": int(train_result["skipped_optimizer_steps"]),
        "gpu_memory_allocated_before": memory_before,
        "gpu_memory_allocated_after": memory_after,
        "gpu_peak_memory_bytes": memory_peak,
        "gpu_total_memory_bytes": total_memory,
        "checkpoint_hashes_unchanged": before_hashes == after_hashes,
        "checkpoint_hashes": before_hashes,
    }
    if not report["passed"]:
        raise RuntimeError(f"Resume smoke test failed: {json.dumps(report, indent=2)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated resume train/validation batch.")
    parser.add_argument("--config", default="configs/training/phase2_5.yaml")
    parser.add_argument("--architecture", default="convnext_tiny")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/training/crop_disease_phase2_5/convnext_tiny/last.pt",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path outside the run bundle.")
    args = parser.parse_args()
    report = run_smoke_test(args.config, args.architecture, args.checkpoint)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
