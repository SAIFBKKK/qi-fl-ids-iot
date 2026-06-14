"""Shared-supernet and tier submodels for P7 HeteroFL."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


MAX_H1 = 256
MAX_H2 = 128
INPUT_DIM = 28
TIER_DIMS = {
    "weak": (64, 0),
    "medium": (128, 64),
    "powerful": (256, 128),
}


class HeteroSuperNet(nn.Module):
    """Full P7 supernet: 28 -> 256 -> 128 -> output_dim."""

    def __init__(self, *, input_dim: int = INPUT_DIM, output_dim: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.fc1 = nn.Linear(self.input_dim, MAX_H1)
        self.fc2 = nn.Linear(MAX_H1, MAX_H2)
        self.fc3 = nn.Linear(MAX_H2, self.output_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)

    def count_parameters(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))


class TierSubNet(nn.Module):
    """Tier submodel. Weak skips fc2 and uses the prefix output head."""

    def __init__(self, *, tier: str, input_dim: int = INPUT_DIM, output_dim: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        if tier not in TIER_DIMS:
            raise ValueError(f"unknown tier: {tier}")
        self.tier = tier
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.h1, self.h2 = TIER_DIMS[tier]
        self.fc1 = nn.Linear(self.input_dim, self.h1)
        if self.h2:
            self.fc2 = nn.Linear(self.h1, self.h2)
            head_in = self.h2
        else:
            self.fc2 = None
            head_in = self.h1
        self.fc3 = nn.Linear(head_in, self.output_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.fc1(x)))
        if self.fc2 is not None:
            x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)

    def count_parameters(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))


def build_supernet(*, output_dim: int, dropout: float = 0.2, input_dim: int = INPUT_DIM) -> HeteroSuperNet:
    return HeteroSuperNet(input_dim=input_dim, output_dim=output_dim, dropout=dropout)


def build_tier_model(*, tier: str, output_dim: int, dropout: float = 0.2, input_dim: int = INPUT_DIM) -> TierSubNet:
    return TierSubNet(tier=tier, input_dim=input_dim, output_dim=output_dim, dropout=dropout)


def architecture_for_tier(tier: str, output_dim: int | str = "output_dim") -> str:
    if tier == "weak":
        return f"28 -> 64 -> {output_dim}"
    if tier == "medium":
        return f"28 -> 128 -> 64 -> {output_dim}"
    if tier == "powerful":
        return f"28 -> 256 -> 128 -> {output_dim}"
    if tier == "supernet":
        return f"28 -> 256 -> 128 -> {output_dim}"
    raise ValueError(f"unknown tier: {tier}")


def model_size_bytes(model_or_state: nn.Module | dict[str, torch.Tensor]) -> int:
    if isinstance(model_or_state, nn.Module):
        tensors = model_or_state.state_dict().values()
    else:
        tensors = model_or_state.values()
    return int(sum(t.detach().cpu().numpy().nbytes for t in tensors))


def tier_parameter_summary(output_dim: int, dropout: float = 0.2) -> dict[str, dict[str, Any]]:
    summary = {}
    for tier in ["weak", "medium", "powerful"]:
        model = build_tier_model(tier=tier, output_dim=output_dim, dropout=dropout)
        summary[tier] = {
            "architecture": architecture_for_tier(tier, output_dim),
            "num_parameters": model.count_parameters(),
            "model_size_bytes": model_size_bytes(model),
        }
    supernet = build_supernet(output_dim=output_dim, dropout=dropout)
    summary["supernet"] = {
        "architecture": architecture_for_tier("supernet", output_dim),
        "num_parameters": supernet.count_parameters(),
        "model_size_bytes": model_size_bytes(supernet),
    }
    return summary
