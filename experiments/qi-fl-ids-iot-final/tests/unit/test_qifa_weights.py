from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qifa.scoring import amplitudes_from_theta, hybrid_weights, normalize_scores_to_theta, probabilities_from_amplitudes


def test_probabilities_sum_to_one() -> None:
    probs = probabilities_from_amplitudes(amplitudes_from_theta(normalize_scores_to_theta([0.1, 0.2, 0.3])))
    assert abs(float(probs.sum()) - 1.0) < 1e-12


def test_gamma_zero_gives_fedavg() -> None:
    fedavg = np.asarray([0.2, 0.3, 0.5], dtype=float)
    probs = np.asarray([0.5, 0.2, 0.3], dtype=float)
    assert np.allclose(hybrid_weights(fedavg, probs, 0.0), fedavg)


def test_gamma_one_gives_qifa_probabilities() -> None:
    fedavg = np.asarray([0.2, 0.3, 0.5], dtype=float)
    probs = np.asarray([0.5, 0.2, 0.3], dtype=float)
    assert np.allclose(hybrid_weights(fedavg, probs, 1.0), probs)


def test_hybrid_weights_non_negative() -> None:
    weights = hybrid_weights([0.2, 0.3, 0.5], [0.5, 0.2, 0.3], 0.75)
    assert np.all(weights >= 0.0)
