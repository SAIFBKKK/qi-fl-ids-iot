"""Tensor-network-inspired low-rank layers.

This is an MPS-style factorization for dense Linear layers: a matrix W is
represented as two smaller factors U and V with rank r.
"""

from __future__ import annotations

import torch
from torch import nn


class LowRankLinear(nn.Module):
    """Linear layer approximated by two factor matrices."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.left = nn.Linear(self.in_features, self.rank, bias=False)
        self.right = nn.Linear(self.rank, self.out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.right(self.left(x))
