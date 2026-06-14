"""Client data loading for P5 FedAvg L1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ClientArrays:
    """In-memory arrays for one client split."""

    X: np.ndarray
    y: np.ndarray
    label_id_original: np.ndarray
    row_id: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.y.shape[0])


def load_client_npz(
    path: Path,
    *,
    max_samples: int | None = None,
    seed: int = 42,
) -> ClientArrays:
    """Load a client NPZ; optionally deterministic-subsample for smoke runs."""

    with np.load(path, allow_pickle=False) as data:
        required = ["X", "y_binary", "label_id_original", "row_id"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"{path} missing arrays: {missing}")
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y_binary"], dtype=np.int64)
        label_id_original = np.asarray(data["label_id_original"], dtype=np.int64)
        row_id = np.asarray(data["row_id"], dtype=np.int64)

    if max_samples is not None and max_samples > 0 and y.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(np.arange(y.shape[0]), size=max_samples, replace=False))
        X = X[indices]
        y = y[indices]
        label_id_original = label_id_original[indices]
        row_id = row_id[indices]

    return ClientArrays(X=X, y=y, label_id_original=label_id_original, row_id=row_id)


def make_dataloader(
    arrays: ClientArrays,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device | None = None,
) -> DataLoader:
    """Build a PyTorch DataLoader."""

    dataset = TensorDataset(torch.from_numpy(arrays.X), torch.from_numpy(arrays.y).long())
    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = bool(device is not None and device.type == "cuda")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=pin_memory,
    )


def binary_counts(y: np.ndarray) -> dict[str, int]:
    """Return normal/attack counts."""

    return {
        "normal_count": int(np.sum(y == 0)),
        "attack_count": int(np.sum(y == 1)),
    }
