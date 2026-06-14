"""Fast tests for QGA chromosome utilities."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga.chromosome import repair_mask_bounds, sample_quantum_mask, theta_to_probabilities


def test_theta_probabilities_are_bounded() -> None:
    theta = np.array([0.0, np.pi / 4, np.pi / 2])
    probs = theta_to_probabilities(theta)
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)
    assert probs[0] == 0
    assert np.isclose(probs[-1], 1)


def test_repair_mask_respects_min_max_features() -> None:
    rng = np.random.default_rng(42)
    mask = np.zeros(28, dtype=np.int8)
    repaired = repair_mask_bounds(mask, min_features=8, max_features=24, rng=rng)
    assert 8 <= int(repaired.sum()) <= 24
    mask = np.ones(28, dtype=np.int8)
    repaired = repair_mask_bounds(mask, min_features=8, max_features=24, rng=rng)
    assert 8 <= int(repaired.sum()) <= 24


def test_sample_quantum_mask_has_correct_shape() -> None:
    rng = np.random.default_rng(42)
    theta = np.full(28, np.pi / 4)
    mask = sample_quantum_mask(theta, rng, min_features=8, max_features=24)
    assert mask.shape == (28,)
    assert set(mask.tolist()).issubset({0, 1})
    assert 8 <= int(mask.sum()) <= 24
