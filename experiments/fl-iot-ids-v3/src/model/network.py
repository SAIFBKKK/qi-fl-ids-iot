from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Lightweight tabular MLP for IoT IDS federated clients.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()

        hidden_dims = tuple(int(dim) for dim in hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer.")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError(f"hidden_dims must be positive, got {hidden_dims!r}.")

        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
