from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedtn_mps.compression import estimate_low_rank_compression


def _config() -> dict:
    return {
        "input_dim": 12,
        "hidden_layers": [128, 64],
        "output_dim": 2,
        "evaluation": {"clients": 3, "rounds": 30},
    }


def test_rank_8_compression_reduces_parameters() -> None:
    result = estimate_low_rank_compression(_config(), rank=8)
    assert result.compressed_num_parameters < result.dense_num_parameters
    assert result.compression_ratio < 1.0


def test_high_rank_can_emit_warning_for_small_model() -> None:
    result = estimate_low_rank_compression(_config(), rank=32)
    assert result.compressed_num_parameters >= result.dense_num_parameters
    assert result.warnings
