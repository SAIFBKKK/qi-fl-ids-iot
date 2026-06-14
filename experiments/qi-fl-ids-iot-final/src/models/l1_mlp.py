"""Centralized L1 binary MLP for P4."""

from __future__ import annotations

import torch
from torch import nn


class CentralizedL1MLP(nn.Module):
    """Configurable 28 -> hidden -> 2 binary classifier."""

    def __init__(
        self,
        input_dim: int = 28,
        hidden_layers: list[int] | tuple[int, ...] = (128, 64),
        output_dim: int = 2,
        dropout: float = 0.2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim != 2:
            raise ValueError("P4 L1 output_dim must be 2")

        activation_layer: type[nn.Module]
        if activation.lower() == "relu":
            activation_layer = nn.ReLU
        else:
            raise ValueError(f"unsupported activation: {activation}")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            if hidden_dim <= 0:
                raise ValueError("hidden layer dimensions must be positive")
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(activation_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [batch, 2]."""

        return self.network(x)

    def count_parameters(self) -> int:
        """Count trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
