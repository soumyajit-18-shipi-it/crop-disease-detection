from __future__ import annotations

from collections import Counter
from typing import Iterable

import torch
from torch import nn


def build_class_weighted_loss(dataset, num_classes: int, device: str) -> nn.CrossEntropyLoss:
    labels = [label for _, label in getattr(dataset, "samples", [])]
    counts = Counter(labels)
    if not counts or min(counts.values()) == max(counts.values()):
        return nn.CrossEntropyLoss()
    total = sum(counts.values())
    weights = torch.ones(num_classes, dtype=torch.float32)
    for class_idx in range(num_classes):
        weights[class_idx] = total / (num_classes * max(counts.get(class_idx, 1), 1))
    return nn.CrossEntropyLoss(weight=weights.to(device))


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    preds = logits.argmax(dim=1)
    return int((preds == labels).sum().item()), int(labels.numel())


def train_one_epoch(model: nn.Module, dataloader: Iterable, optimizer, criterion, device: str) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        batch_correct, batch_total = _accuracy(logits, labels)
        correct += batch_correct
        total += batch_total
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate(model: nn.Module, dataloader: Iterable, criterion, device: str) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions: list[int] = []
    labels_out: list[int] = []
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        batch_correct, batch_total = _accuracy(logits, labels)
        correct += batch_correct
        total += batch_total
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        labels_out.extend(labels.cpu().tolist())
    return total_loss / max(total, 1), correct / max(total, 1), predictions, labels_out
