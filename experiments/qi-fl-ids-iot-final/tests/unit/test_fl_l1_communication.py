from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fl_l1.communication import human_readable_bytes, model_size_bytes, round_bandwidth  # noqa: E402


def test_model_size_bytes_positive() -> None:
    state = {"w": torch.zeros(2, 3), "b": torch.ones(3)}
    assert model_size_bytes(state) > 0


def test_round_bandwidth_formula() -> None:
    result = round_bandwidth(model_size_bytes_value=100, num_clients=3, previous_cumulative_bytes=50)
    assert result["upload_bytes"] == 300
    assert result["download_bytes"] == 300
    assert result["total_bytes"] == 600
    assert result["cumulative_bytes"] == 650


def test_bandwidth_cumulative_formula() -> None:
    cumulative = 0
    for _ in range(30):
        result = round_bandwidth(
            model_size_bytes_value=48_392,
            num_clients=3,
            previous_cumulative_bytes=cumulative,
        )
        cumulative = int(result["cumulative_bytes"])
    assert cumulative == 8_710_560


def test_human_readable_bytes() -> None:
    assert human_readable_bytes(512) == "512.00 B"
    assert human_readable_bytes(2048) == "2.00 KB"
