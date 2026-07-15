from __future__ import annotations

import copy
from collections import Counter
from contextlib import nullcontext
from typing import Iterable

import numpy as np
import torch
from torch import nn


class ModelEMA:
    """Exponential moving average with a short bias-reducing warm start."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        self.num_updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            model_value = source[name].detach()
            if value.is_floating_point():
                value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                value.copy_(model_value)

    def load_state_dict(self, state_dict: dict, num_updates: int = 0) -> None:
        self.module.load_state_dict(state_dict)
        self.num_updates = int(num_updates)


def _dataset_labels(dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        return [int(label) for label in dataset.targets]
    return [int(sample[1]) for sample in getattr(dataset, "samples", [])]


def build_class_weights(
    dataset,
    num_classes: int,
    strategy: str = "effective_num",
    effective_num_beta: float = 0.9999,
) -> torch.Tensor | None:
    counts = Counter(_dataset_labels(dataset))
    if not counts:
        return None
    missing = [class_index for class_index in range(num_classes) if counts[class_index] == 0]
    if missing:
        raise ValueError(f"Training split has no samples for class indices: {missing}")
    count_tensor = torch.tensor([counts[index] for index in range(num_classes)], dtype=torch.float64)
    if strategy == "inverse_frequency":
        weights = count_tensor.sum() / (num_classes * count_tensor)
    elif strategy == "effective_num":
        beta = float(effective_num_beta)
        weights = (1.0 - beta) / (1.0 - torch.pow(beta, count_tensor))
        weights /= weights.mean()
    else:
        raise ValueError(f"Unknown class-weight strategy: {strategy}")
    return weights.to(dtype=torch.float32)


def build_class_weighted_loss(
    dataset,
    num_classes: int,
    device: str,
    label_smoothing: float = 0.0,
    strategy: str = "effective_num",
    effective_num_beta: float = 0.9999,
) -> nn.CrossEntropyLoss:
    weights = build_class_weights(dataset, num_classes, strategy, effective_num_beta)
    if weights is not None:
        weights = weights.to(device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


def _autocast(device: str, enabled: bool, amp_dtype: str):
    if not enabled or not device.startswith("cuda"):
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _cutmix_box(height: int, width: int, lam: float):
    cut_ratio = float(np.sqrt(1.0 - lam))
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)
    center_x = int(np.random.randint(0, width))
    center_y = int(np.random.randint(0, height))
    x1 = max(center_x - cut_width // 2, 0)
    x2 = min(center_x + cut_width // 2, width)
    y1 = max(center_y - cut_height // 2, 0)
    y2 = min(center_y + cut_height // 2, height)
    return x1, y1, x2, y2


def apply_batch_augmentation(images: torch.Tensor, labels: torch.Tensor, augmentation):
    """Apply at most one of MixUp or CutMix and return soft-target parts."""
    if augmentation is None or not augmentation.enabled or images.size(0) < 2:
        return images, labels, labels, 1.0, "none"
    mixup_p = float(augmentation.mixup_probability)
    cutmix_p = float(augmentation.cutmix_probability)
    draw = float(np.random.random())
    if draw >= mixup_p + cutmix_p:
        return images, labels, labels, 1.0, "none"
    permutation = torch.randperm(images.size(0), device=images.device)
    paired_labels = labels[permutation]
    if draw < mixup_p and augmentation.mixup_alpha > 0:
        lam = float(np.random.beta(augmentation.mixup_alpha, augmentation.mixup_alpha))
        mixed = images.mul(lam).add(images[permutation], alpha=1.0 - lam)
        return mixed, labels, paired_labels, lam, "mixup"
    if augmentation.cutmix_alpha <= 0:
        return images, labels, labels, 1.0, "none"
    lam = float(np.random.beta(augmentation.cutmix_alpha, augmentation.cutmix_alpha))
    x1, y1, x2, y2 = _cutmix_box(images.shape[-2], images.shape[-1], lam)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.shape[-1] * images.shape[-2]))
    return mixed, labels, paired_labels, float(lam), "cutmix"


def run_epoch(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: str,
    optimizer=None,
    scaler=None,
    scheduler=None,
    ema: ModelEMA | None = None,
    mixed_precision: bool = False,
    gradient_clip_norm: float = 0.0,
    gradient_accumulation_steps: int = 1,
    augmentation=None,
    amp_dtype: str = "float16",
) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0.0
    total = 0
    predictions: list[int] = []
    targets: list[int] = []
    all_logits: list[list[float]] = []
    mix_counts = Counter()
    accumulation_steps = max(int(gradient_accumulation_steps), 1)
    num_batches = len(dataloader)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        if training:
            optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                images, labels_a, labels_b, lam, mix_name = apply_batch_augmentation(
                    images, labels, augmentation
                )
                mix_counts[mix_name] += 1
            else:
                labels_a = labels_b = labels
                lam = 1.0
            with _autocast(device, mixed_precision, amp_dtype):
                logits = model(images)
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
            if training:
                group_start = (batch_index // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, num_batches - group_start)
                scaled_loss = loss / group_size
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == num_batches
                if should_step:
                    if scaler is not None and scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    if gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    optimizer_updated = True
                    if scaler is not None and scaler.is_enabled():
                        scale_before = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer_updated = scaler.get_scale() >= scale_before
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # AMP can skip optimizer.step on overflow. Advancing LR/EMA
                    # in that case both triggers warnings and desynchronizes state.
                    if optimizer_updated:
                        if scheduler is not None:
                            scheduler.step()
                        if ema is not None:
                            ema.update(model)
            predicted = logits.argmax(1)
            total_loss += float(loss.detach().item()) * labels.size(0)
            correct += lam * float((predicted == labels_a).sum().item())
            correct += (1.0 - lam) * float((predicted == labels_b).sum().item())
            total += labels.numel()
            if not training:
                predictions.extend(predicted.detach().cpu().tolist())
                targets.extend(labels.detach().cpu().tolist())
                all_logits.extend(logits.detach().float().cpu().tolist())
    probabilities = (
        torch.softmax(torch.tensor(all_logits, dtype=torch.float32), dim=1).tolist()
        if all_logits
        else []
    )
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "predictions": predictions,
        "targets": targets,
        "logits": all_logits,
        "probabilities": probabilities,
        "batch_augmentation_counts": dict(mix_counts),
    }
