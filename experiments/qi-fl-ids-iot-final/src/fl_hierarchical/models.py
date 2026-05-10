"""PyTorch models for P6 hierarchical L2/L3 experiments."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from torch import nn


class HierarchicalMLP(nn.Module):
    """Configurable MLP for L2 family and L3 attack-type classification."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        dropout: float,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation.lower() != "relu":
            raise ValueError(f"Unsupported activation for P6: {activation}")
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for hidden in hidden_layers:
            layers.append(nn.Linear(previous, int(hidden)))
            layers.append(nn.ReLU())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            previous = int(hidden)
        layers.append(nn.Linear(previous, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad))


def build_model(model_cfg: dict[str, Any], *, output_dim: int) -> HierarchicalMLP:
    """Build a P6 model from YAML config."""

    return HierarchicalMLP(
        input_dim=int(model_cfg["input_dim"]),
        hidden_layers=[int(item) for item in model_cfg["hidden_layers"]],
        output_dim=int(output_dim),
        dropout=float(model_cfg["dropout"]),
        activation=str(model_cfg.get("activation", "relu")),
    )


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [value.detach().cpu().numpy() for _, value in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    if len(keys) != len(parameters):
        raise ValueError(f"parameter length mismatch: expected {len(keys)}, got {len(parameters)}")
    state_dict = OrderedDict(
        (key, torch.tensor(array, dtype=model.state_dict()[key].dtype))
        for key, array in zip(keys, parameters)
    )
    model.load_state_dict(state_dict, strict=True)


def model_size_bytes(model_or_params: nn.Module | list[np.ndarray]) -> int:
    if isinstance(model_or_params, list):
        return int(sum(np.asarray(array).nbytes for array in model_or_params))
    return int(sum(tensor.detach().cpu().numpy().nbytes for tensor in model_or_params.state_dict().values()))
