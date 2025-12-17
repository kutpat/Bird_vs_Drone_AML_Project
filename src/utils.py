# src/utils.py
from __future__ import annotations
import os
import random

from typing import Dict, Tuple, Optional
import torch
import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

def get_num_workers(default: int = 4) -> int:
    """Safe default for num_workers across OSes."""
    # On Windows, too many workers can be unstable sometimes.
    return int(os.environ.get("NUM_WORKERS", default))

@torch.no_grad()
def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

@torch.no_grad()
def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Binary classification metrics without sklearn.
    Assumes labels are 0/1.
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    acc = float((y_true == y_pred).mean())

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }