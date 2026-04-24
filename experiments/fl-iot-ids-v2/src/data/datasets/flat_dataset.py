from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class FlatTensorDataset(Dataset):
    """In-memory tensor dataset used by tests and hierarchical wrappers."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} != {len(y)}")
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class IoTLocalDataset(Dataset):
    """
    PyTorch Dataset for local IoT node data.

    Loads a .npz file produced by the preprocessing step.

    Expected NPZ structure:
        X : float32 array [n_samples, n_features]
        y : int64 array   [n_samples]
    """

    def __init__(self, npz_path: str | Path):
        self.npz_path = Path(npz_path)

        if not self.npz_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.npz_path}")

        data = np.load(self.npz_path, allow_pickle=True)

        if "X" not in data or "y" not in data:
            raise KeyError(
                f"Invalid NPZ format in {self.npz_path}. "
                "Expected arrays: 'X' and 'y'."
            )

        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.long)

        self.feature_names = data.get("feature_names", None)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_npz_arrays(npz_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Utility loader for analyses/scripts that need raw NumPy arrays
    instead of a PyTorch Dataset.
    """
    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    if "X" not in data or "y" not in data:
        raise KeyError(
            f"Invalid NPZ format in {npz_path}. "
            "Expected arrays: 'X' and 'y'."
        )

    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)

    return X, y


def load_npz_xy(npz_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility alias for older hierarchical code."""
    return load_npz_arrays(npz_path)
