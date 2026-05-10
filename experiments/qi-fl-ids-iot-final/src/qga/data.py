"""Data loading helpers for P8 QGA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class L1Arrays:
    X: np.ndarray
    y: np.ndarray
    label_id_original: np.ndarray
    row_id: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.y.shape[0])


def load_l1_npz(path: str | Path) -> L1Arrays:
    payload = np.load(path)
    required = {"X", "y_binary", "label_id_original", "row_id"}
    missing = required.difference(payload.files)
    if missing:
        raise ValueError(f"{path} missing NPZ keys: {sorted(missing)}")
    return L1Arrays(
        X=payload["X"].astype(np.float32, copy=False),
        y=payload["y_binary"].astype(np.int64, copy=False),
        label_id_original=payload["label_id_original"].astype(np.int64, copy=False),
        row_id=payload["row_id"],
    )


def stratified_sample_indices(
    y: np.ndarray,
    *,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    y = np.asarray(y)
    if max_samples is None or int(max_samples) <= 0 or y.shape[0] <= int(max_samples):
        return np.arange(y.shape[0])
    rng = np.random.default_rng(int(seed))
    classes = np.unique(y)
    per_class = max(1, int(max_samples) // max(len(classes), 1))
    chosen: list[np.ndarray] = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        take = min(idx.shape[0], per_class)
        chosen.append(rng.choice(idx, size=take, replace=False))
    merged = np.concatenate(chosen)
    if merged.shape[0] < int(max_samples):
        remaining = np.setdiff1d(np.arange(y.shape[0]), merged, assume_unique=False)
        extra_count = min(int(max_samples) - merged.shape[0], remaining.shape[0])
        if extra_count > 0:
            extra = rng.choice(remaining, size=extra_count, replace=False)
            merged = np.concatenate([merged, extra])
    rng.shuffle(merged)
    return merged.astype(np.int64)


def sample_arrays(arrays: L1Arrays, *, max_samples: int | None, seed: int) -> L1Arrays:
    indices = stratified_sample_indices(arrays.y, max_samples=max_samples, seed=seed)
    return L1Arrays(
        X=arrays.X[indices],
        y=arrays.y[indices],
        label_id_original=arrays.label_id_original[indices],
        row_id=arrays.row_id[indices],
    )


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), generator=generator)


def load_qga_train_val(config: dict[str, Any], *, mode: str) -> tuple[L1Arrays, L1Arrays]:
    from qga.config import qga_params, repo_path

    params = qga_params(config, mode=mode)
    max_samples = int(params["max_samples_for_fitness"])
    train = sample_arrays(load_l1_npz(repo_path(config, "inputs.train_npz")), max_samples=max_samples, seed=params["seed"])
    val = sample_arrays(load_l1_npz(repo_path(config, "inputs.val_npz")), max_samples=max_samples, seed=params["seed"] + 1)
    return train, val
