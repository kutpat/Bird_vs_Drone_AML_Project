# src/train.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from snntorch import spikegen

from .utils import compute_binary_metrics

@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    device: str = "cpu"
    lambda_spikes: float = 0.0  # optional: spike penalty


def encode_rate(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    x is normalized to [-1,1]. Convert back to [0,1] for rate coding.
    Returns spikes [T,B,C,H,W]
    """
    x01 = (x + 1.0) / 2.0
    x01 = torch.clamp(x01, 0.0, 1.0)
    return spikegen.rate(x01, num_steps=T)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: Optimizer,
    criterion: nn.Module,
    T: int,
    device: torch.device,
    lambda_spikes: float = 0.0,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_spikes = 0.0

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        x_spk = encode_rate(x, T=T)

        optimizer.zero_grad()
        logits, stats = model(x_spk)

        loss = criterion(logits, y)
        if lambda_spikes > 0.0:
            loss = loss + lambda_spikes * stats["avg_spikes_per_sample"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)
        total_spikes += stats["total_spikes"].item()

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
        "avg_spikes_per_sample": (total_spikes / max(1, total_samples)),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    T: int,
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_spikes = 0.0

    all_y = []
    all_pred = []

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        x_spk = encode_rate(x, T=T)

        logits, stats = model(x_spk)
        loss = criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)
        total_spikes += stats["total_spikes"].item()

        all_y.append(y.detach().cpu())
        all_pred.append(preds.detach().cpu())

    y_true = torch.cat(all_y, dim=0).numpy()
    y_pred = torch.cat(all_pred, dim=0).numpy()

    m = compute_binary_metrics(y_true, y_pred)

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": m["accuracy"],
        "f1": m["f1"],
        "avg_spikes_per_sample": (total_spikes / max(1, total_samples)),
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    loader,
    T: int,
    device: torch.device,
    max_batches: int | None = None,
):
    """
    Returns y_true (np), y_pred (np), avg_spikes_per_sample (float)
    """
    model.eval()

    total_samples = 0
    total_spikes = 0.0

    all_y = []
    all_pred = []

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        x_spk = encode_rate(x, T=T)

        logits, stats = model(x_spk)
        preds = torch.argmax(logits, dim=1)

        total_samples += y.size(0)
        total_spikes += stats["total_spikes"].item()

        all_y.append(y.detach().cpu())
        all_pred.append(preds.detach().cpu())

    y_true = torch.cat(all_y, dim=0).numpy()
    y_pred = torch.cat(all_pred, dim=0).numpy()
    avg_spikes = total_spikes / max(1, total_samples)

    return y_true, y_pred, avg_spikes



