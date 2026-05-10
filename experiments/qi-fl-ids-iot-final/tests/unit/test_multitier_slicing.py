from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multitier_heterofl.slicing import contribution_indices, extract_tier_state, load_tier_state  # noqa: E402
from multitier_heterofl.supernet import build_supernet, build_tier_model  # noqa: E402


def test_weak_slice_shapes() -> None:
    state = extract_tier_state(build_supernet(output_dim=2).state_dict(), "weak")
    assert state["fc1.weight"].shape == (64, 28)
    assert "fc2.weight" not in state
    assert state["fc3.weight"].shape == (2, 64)


def test_medium_slice_loads() -> None:
    state = extract_tier_state(build_supernet(output_dim=8).state_dict(), "medium")
    model = build_tier_model(tier="medium", output_dim=8)
    load_tier_state(model, state)
    assert model.fc2 is not None


def test_contribution_indices_skip_fc2_for_weak() -> None:
    assert "fc2.weight" not in contribution_indices("weak")
    assert "fc2.weight" in contribution_indices("medium")
