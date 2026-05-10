from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multitier_heterofl.supernet import build_supernet, build_tier_model, tier_parameter_summary  # noqa: E402


def test_supernet_forward_l1_shape() -> None:
    model = build_supernet(output_dim=2, dropout=0.0)
    assert model(torch.zeros(4, 28)).shape == (4, 2)


def test_tier_forward_shapes_l2() -> None:
    for tier in ["weak", "medium", "powerful"]:
        model = build_tier_model(tier=tier, output_dim=8, dropout=0.0)
        assert model(torch.zeros(4, 28)).shape == (4, 8)


def test_parameter_order_by_tier() -> None:
    summary = tier_parameter_summary(2)
    assert summary["weak"]["num_parameters"] < summary["medium"]["num_parameters"] < summary["powerful"]["num_parameters"]
