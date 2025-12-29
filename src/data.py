# src/data.py
# Data loading pipeline:
# - loads images from data/bird and data/drone (ImageFolder)
# - applies augmentation on train only
# - creates a deterministic *stratified* train/val/test split
# - returns PyTorch DataLoaders + class names
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .utils import set_seed, get_num_workers


@dataclass
class DataConfig:
    data_dir: str = "data"
    image_size: int = 64
    batch_size: int = 64
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True


def _build_transforms(image_size: int):
    """
       Train transforms include augmentation to reduce overfitting.
       Val/Test transforms are deterministic (no augmentation).
       Both use the same normalization as in training + Streamlit inference.
       """
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        # Simple normalization for images in [0,1]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    return train_tf, eval_tf


def _stratified_split_indices(
    targets: List[int],
    val_split: float,
    test_split: float,
    seed: int
) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified split for classification.

    We split per class so that bird/drone ratios stay similar in train/val/test.
    Returns lists of indices: train_idx, val_idx, test_idx.
    """
    assert 0.0 <= val_split < 1.0
    assert 0.0 <= test_split < 1.0
    assert val_split + test_split < 1.0

    # Torch Generator so that shuffling is reproducible
    g = torch.Generator().manual_seed(seed)

    targets_t = torch.tensor(targets, dtype=torch.long)
    classes = torch.unique(targets_t).tolist()

    train_idx, val_idx, test_idx = [], [], []

    for c in classes:
        idx_c = torch.where(targets_t == c)[0]

        # Deterministic shuffle within this class
        perm = idx_c[torch.randperm(len(idx_c), generator=g)]

        n = len(perm)
        n_test = int(round(n * test_split))
        n_val = int(round(n * val_split))

        test_part = perm[:n_test]
        val_part = perm[n_test:n_test + n_val]
        train_part = perm[n_test + n_val:]

        train_idx += train_part.tolist()
        val_idx += val_part.tolist()
        test_idx += test_part.tolist()

    # Shuffle combined lists (still deterministic)
    train_idx = torch.tensor(train_idx)[torch.randperm(len(train_idx), generator=g)].tolist()
    val_idx = torch.tensor(val_idx)[torch.randperm(len(val_idx), generator=g)].tolist()
    test_idx = torch.tensor(test_idx)[torch.randperm(len(test_idx), generator=g)].tolist()

    return train_idx, val_idx, test_idx


def get_dataloaders(cfg: DataConfig):
    """
    Creates datasets + stratified split + DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    set_seed(cfg.seed)

    if cfg.num_workers == 0:
        # good default on Windows is often 0..2;
        cfg.num_workers = get_num_workers(default=2)

    train_tf, eval_tf = _build_transforms(cfg.image_size)

    # We create two ImageFolder datasets pointing to the same files,
    # but with different transforms (augmented train, clean eval).
    # ImageFolder keeps a stable file ordering, so indices match.
    ds_train = datasets.ImageFolder(root=cfg.data_dir, transform=train_tf)
    ds_eval = datasets.ImageFolder(root=cfg.data_dir, transform=eval_tf)

    targets = ds_train.targets  # same order for ds_eval
    train_idx, val_idx, test_idx = _stratified_split_indices(
        targets=targets,
        val_split=cfg.val_split,
        test_split=cfg.test_split,
        seed=cfg.seed
    )

    # Use Subset to create split datasets without copying files
    train_set = Subset(ds_train, train_idx)
    val_set = Subset(ds_eval, val_idx)
    test_set = Subset(ds_eval, test_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )

    class_names = ds_train.classes  # e.g. ["bird", "drone"] (alphabetical)
    return train_loader, val_loader, test_loader, class_names
