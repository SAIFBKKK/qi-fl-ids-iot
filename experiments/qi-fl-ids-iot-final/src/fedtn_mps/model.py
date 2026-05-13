"""L1 MLP models for P11 compression."""

from __future__ import annotations

import torch
from torch import nn

from .mps_layers import LowRankLinear


class L1QGAMLP(nn.Module):
    """Reference dense L1 QGA model: input -> 128 -> 64 -> 2."""

    def __init__(self, input_dim: int = 12, hidden_layers: list[int] | None = None, output_dim: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        hidden = hidden_layers or [128, 64]
        self.fc1 = nn.Linear(int(input_dim), int(hidden[0]))
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(hidden[0]), int(hidden[1]))
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(float(dropout))
        self.fc3 = nn.Linear(int(hidden[1]), int(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        return self.fc3(x)


class MPSCompressedL1MLP(nn.Module):
    """Low-rank compressed L1 QGA model."""

    def __init__(
        self,
        input_dim: int = 12,
        hidden_layers: list[int] | None = None,
        output_dim: int = 2,
        rank: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden = hidden_layers or [128, 64]
        self.fc1 = LowRankLinear(int(input_dim), int(hidden[0]), int(rank))
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(float(dropout))
        self.fc2 = LowRankLinear(int(hidden[0]), int(hidden[1]), int(rank))
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(float(dropout))
        self.fc3 = nn.Linear(int(hidden[1]), int(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        return self.fc3(x)


def build_dense_model(config: dict, *, input_dim: int | None = None) -> L1QGAMLP:
    return L1QGAMLP(
        input_dim=int(input_dim if input_dim is not None else config.get("input_dim", 12)),
        hidden_layers=list(config.get("hidden_layers", [128, 64])),
        output_dim=int(config.get("output_dim", 2)),
    )


def build_compressed_model(config: dict, rank: int, *, input_dim: int | None = None) -> MPSCompressedL1MLP:
    return MPSCompressedL1MLP(
        input_dim=int(input_dim if input_dim is not None else config.get("input_dim", 12)),
        hidden_layers=list(config.get("hidden_layers", [128, 64])),
        output_dim=int(config.get("output_dim", 2)),
        rank=int(rank),
    )
