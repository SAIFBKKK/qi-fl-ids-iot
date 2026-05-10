from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fl_l1.aggregation import fedavg_state_dicts  # noqa: E402


def test_fedavg_weighted_average_two_clients() -> None:
    state_a = {"w": torch.tensor([1.0, 3.0])}
    state_b = {"w": torch.tensor([5.0, 7.0])}
    result = fedavg_state_dicts([state_a, state_b], [1, 3], client_ids=["a", "b"])
    assert torch.allclose(result.state_dict["w"], torch.tensor([4.0, 6.0]))
    assert result.weights == {"a": 0.25, "b": 0.75}


def test_fedavg_preserves_tensor_shapes() -> None:
    state_a = {"w": torch.ones(2, 3), "b": torch.zeros(3)}
    state_b = {"w": torch.zeros(2, 3), "b": torch.ones(3)}
    result = fedavg_state_dicts([state_a, state_b], [2, 2])
    assert result.state_dict["w"].shape == state_a["w"].shape
    assert result.state_dict["b"].shape == state_a["b"].shape


def test_fedavg_rejects_mismatched_shapes() -> None:
    state_a = {"w": torch.ones(2, 3)}
    state_b = {"w": torch.ones(3, 2)}
    with pytest.raises(ValueError, match="shape mismatch"):
        fedavg_state_dicts([state_a, state_b], [1, 1])
