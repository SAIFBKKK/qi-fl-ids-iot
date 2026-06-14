from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qifa.aggregation import aggregate_weighted_ndarrays, parameter_drift


def test_qifa_weighted_aggregation_matches_manual_average() -> None:
    params = [
        [np.asarray([1.0, 3.0], dtype=np.float32)],
        [np.asarray([5.0, 7.0], dtype=np.float32)],
    ]
    aggregated = aggregate_weighted_ndarrays(params, [0.25, 0.75])
    assert np.allclose(aggregated[0], np.asarray([4.0, 6.0], dtype=np.float32))


def test_drift_penalty_observes_larger_divergence() -> None:
    reference = [np.asarray([0.0, 0.0], dtype=np.float32)]
    near = [np.asarray([0.1, 0.1], dtype=np.float32)]
    far = [np.asarray([2.0, 2.0], dtype=np.float32)]
    assert parameter_drift(reference, far) > parameter_drift(reference, near)
