from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class IoTLocalDataset(Dataset):
    """
    PyTorch Dataset for local IoT node data.

    Loads a .npz file produced by the preprocessing step.

    Expected NPZ structure:
        X : float32 array [n_samples, n_features]
        y : int64 array   [n_samples]
    """

    def __init__(
        self,
        npz_path: str | Path,
        selected_feature_indices: list[int] | None = None,
    ):
        self.npz_path = Path(npz_path)

        if not self.npz_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.npz_path}")

        data = np.load(self.npz_path, allow_pickle=True)

        X = data["X"]
        feature_names = data.get("feature_names", None)
        if selected_feature_indices is not None:
            indices = np.asarray(selected_feature_indices, dtype=int)
            if indices.ndim != 1 or indices.size == 0:
                raise ValueError("selected_feature_indices must be a non-empty 1D list.")
            if indices.min() < 0 or indices.max() >= X.shape[1]:
                raise ValueError(
                    f"Selected feature index out of bounds for X.shape={X.shape}: "
                    f"{indices.tolist()}"
                )
            X = X[:, indices]
            if feature_names is not None:
                feature_names = feature_names[indices]

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.long)

        self.feature_names = feature_names

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
