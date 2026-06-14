from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fl_hierarchical.models import HierarchicalMLP, get_parameters, set_parameters  # noqa: E402


def test_l2_model_forward_shape() -> None:
    model = HierarchicalMLP(input_dim=28, hidden_layers=[128, 64], output_dim=8, dropout=0.2)
    logits = model(torch.zeros(4, 28))
    assert logits.shape == (4, 8)


def test_l3_model_forward_shape() -> None:
    model = HierarchicalMLP(input_dim=28, hidden_layers=[128, 64], output_dim=33, dropout=0.2)
    logits = model(torch.zeros(4, 28))
    assert logits.shape == (4, 33)


def test_parameter_round_trip() -> None:
    model = HierarchicalMLP(input_dim=28, hidden_layers=[128, 64], output_dim=8, dropout=0.2)
    params = get_parameters(model)
    set_parameters(model, params)
    assert len(get_parameters(model)) == len(params)


def test_parameter_count_positive() -> None:
    model = HierarchicalMLP(input_dim=28, hidden_layers=[128, 64], output_dim=8, dropout=0.2)
    assert model.count_parameters() > 0
