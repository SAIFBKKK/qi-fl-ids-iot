from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedtn_mps.model import build_dense_model
from fedtn_mps.size import bandwidth_bytes, compression_ratio, count_parameters, model_size_bytes


def test_model_size_bytes_calculated_from_parameters() -> None:
    model = build_dense_model({"input_dim": 12, "hidden_layers": [128, 64], "output_dim": 2})
    assert count_parameters(model) == 10050
    assert model_size_bytes(model) == 40200


def test_compression_ratio_formula() -> None:
    assert compression_ratio(100, 400) == 0.25


def test_bandwidth_formula() -> None:
    assert bandwidth_bytes(40200, clients=3, rounds=30) == 7236000
