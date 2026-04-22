from __future__ import annotations

from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_class_weights(weights_path: str | Path, device: torch.device | str = "cpu"):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        return None

    with weights_path.open("rb") as f:
        weights = pickle.load(f)

    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        return (((1.0 - pt) ** self.gamma) * ce_loss).mean()


def build_loss(
    class_weights: torch.Tensor | None = None,
    imbalance_strategy: str = "class_weights",
    focal_gamma: float = 2.0,
) -> nn.Module:
    imbalance_strategy = str(imbalance_strategy).lower()

    if imbalance_strategy == "focal_loss":
        return FocalLoss(gamma=focal_gamma)

    if imbalance_strategy == "focal_loss_weighted":
        return FocalLoss(gamma=focal_gamma, alpha=class_weights)

    if imbalance_strategy == "class_weights" and class_weights is not None:
        return nn.CrossEntropyLoss(weight=class_weights)

    return nn.CrossEntropyLoss()
