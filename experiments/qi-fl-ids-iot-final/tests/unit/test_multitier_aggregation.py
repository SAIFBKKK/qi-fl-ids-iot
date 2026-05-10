from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multitier_heterofl.aggregation import aggregate_slice_weighted  # noqa: E402
from multitier_heterofl.slicing import extract_tier_state  # noqa: E402
from multitier_heterofl.supernet import build_supernet  # noqa: E402


def _filled_state(output_dim: int, value: float):
    model = build_supernet(output_dim=output_dim, dropout=0.0)
    return {key: torch.full_like(tensor, value, dtype=torch.float32) for key, tensor in model.state_dict().items()}


def test_single_powerful_replaces_all_slices() -> None:
    global_state = _filled_state(2, 0.0)
    update = extract_tier_state(_filled_state(2, 2.0), "powerful")
    result, info = aggregate_slice_weighted([{"state_dict": update, "num_examples": 10, "tier": "powerful"}], global_state)
    assert torch.allclose(result["fc1.weight"], torch.full_like(result["fc1.weight"], 2.0))
    assert info["updated_ratio"] == 1.0


def test_weak_keeps_uncovered_fc2() -> None:
    global_state = _filled_state(2, 5.0)
    update = extract_tier_state(_filled_state(2, 1.0), "weak")
    result, _ = aggregate_slice_weighted([{"state_dict": update, "num_examples": 10, "tier": "weak"}], global_state)
    assert torch.allclose(result["fc2.weight"], torch.full_like(result["fc2.weight"], 5.0))
    assert torch.allclose(result["fc1.weight"][:64], torch.full_like(result["fc1.weight"][:64], 1.0))


def test_mixed_tiers_weighted_prefix() -> None:
    global_state = _filled_state(2, 0.0)
    weak = extract_tier_state(_filled_state(2, 1.0), "weak")
    medium = extract_tier_state(_filled_state(2, 3.0), "medium")
    result, _ = aggregate_slice_weighted(
        [
            {"state_dict": weak, "num_examples": 100, "tier": "weak"},
            {"state_dict": medium, "num_examples": 300, "tier": "medium"},
        ],
        global_state,
    )
    expected = (100 * 1.0 + 300 * 3.0) / 400
    assert torch.allclose(result["fc1.weight"][:64], torch.full_like(result["fc1.weight"][:64], expected))
