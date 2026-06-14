from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperNet(nn.Module):
    """Nested MLP where smaller tier models are static slices of the full model."""

    MAX_HIDDEN_1 = 256
    MAX_HIDDEN_2 = 128
    INPUT_DIM = 28
    OUTPUT_DIM = 34
    SUPPORTED_WIDTHS = (0.25, 0.5, 1.0)

    def __init__(
        self,
        width: float = 1.0,
        dropout: float = 0.2,
        input_dim: int = INPUT_DIM,
        output_dim: int = OUTPUT_DIM,
        max_hidden_1: int = MAX_HIDDEN_1,
        max_hidden_2: int = MAX_HIDDEN_2,
    ) -> None:
        width = float(width)
        assert width in self.SUPPORTED_WIDTHS, "width must be 0.25, 0.5, or 1.0"
        super().__init__()
        self.width = width
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.max_hidden_1 = int(max_hidden_1)
        self.max_hidden_2 = int(max_hidden_2)
        self.h1 = int(self.max_hidden_1 * self.width)
        self.h2 = int(self.max_hidden_2 * self.width)

        self.fc1 = nn.Linear(self.input_dim, self.h1)
        self.fc2 = nn.Linear(self.h1, self.h2)
        self.fc3 = nn.Linear(self.h2, self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


def _hidden_dims_for_width(target_width: float) -> tuple[int, int]:
    width = float(target_width)
    assert width in SuperNet.SUPPORTED_WIDTHS, "width must be 0.25, 0.5, or 1.0"
    return (
        int(SuperNet.MAX_HIDDEN_1 * width),
        int(SuperNet.MAX_HIDDEN_2 * width),
    )


def extract_submodel_state(
    global_state: Mapping[str, torch.Tensor],
    target_width: float,
) -> dict[str, torch.Tensor]:
    """Extract a static HeteroFL slice from a full-width SuperNet state_dict."""
    h1, h2 = _hidden_dims_for_width(target_width)
    return {
        "fc1.weight": global_state["fc1.weight"][:h1, :].detach().clone(),
        "fc1.bias": global_state["fc1.bias"][:h1].detach().clone(),
        "fc2.weight": global_state["fc2.weight"][:h2, :h1].detach().clone(),
        "fc2.bias": global_state["fc2.bias"][:h2].detach().clone(),
        "fc3.weight": global_state["fc3.weight"][:, :h2].detach().clone(),
        "fc3.bias": global_state["fc3.bias"].detach().clone(),
    }


def load_submodel(model: SuperNet, state_dict: Mapping[str, torch.Tensor]) -> SuperNet:
    model.load_state_dict(dict(state_dict), strict=True)
    return model


def count_parameters(width: float) -> int:
    model = SuperNet(width=width)
    return sum(param.numel() for param in model.parameters())
