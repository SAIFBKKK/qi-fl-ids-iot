from __future__ import annotations

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
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()

        h1, h2 = hidden_dims

        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)