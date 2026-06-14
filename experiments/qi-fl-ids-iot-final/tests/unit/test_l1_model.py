from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.l1_mlp import CentralizedL1MLP  # noqa: E402


def test_model_forward_shape() -> None:
    model = CentralizedL1MLP(input_dim=28, hidden_layers=[128, 64], output_dim=2)
    x = torch.zeros(8, 28)
    logits = model(x)
    assert logits.shape == (8, 2)


def test_model_output_dim_is_2() -> None:
    model = CentralizedL1MLP(input_dim=28, hidden_layers=[128, 64], output_dim=2)
    last_linear = [module for module in model.modules() if isinstance(module, torch.nn.Linear)][-1]
    assert last_linear.out_features == 2


def test_model_parameter_count_positive() -> None:
    model = CentralizedL1MLP(input_dim=28, hidden_layers=[128, 64], output_dim=2)
    assert model.count_parameters() > 0
