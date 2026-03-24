from __future__ import annotations

from pathlib import Path
import pickle

import torch
import torch.nn as nn


def load_class_weights(weights_path: str | Path, device: torch.device | str = "cpu"):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        return None

    with weights_path.open("rb") as f:
        weights = pickle.load(f)

    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_loss(class_weights=None) -> nn.Module:
    if class_weights is not None:
        return nn.CrossEntropyLoss(weight=class_weights)
    return nn.CrossEntropyLoss()