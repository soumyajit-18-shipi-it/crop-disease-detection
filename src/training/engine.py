from typing import Iterable

import torch
from torch import nn


def train_one_epoch(model: nn.Module, dataloader: Iterable, optimizer, criterion, device: str) -> float:
    """Train one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def validate(model: nn.Module, dataloader: Iterable, criterion, device: str) -> float:
    """Validate one epoch and return average loss."""
    model.eval()
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        total_loss += criterion(model(images), labels).item()
    return total_loss / max(len(dataloader), 1)
