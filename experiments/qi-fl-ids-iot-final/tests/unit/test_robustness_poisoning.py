from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robustness.poisoning import apply_attack_to_normal, apply_feature_noise, apply_label_flip


def test_label_flip_respects_poison_rate() -> None:
    X = np.zeros((10, 2), dtype=np.float32)
    y = np.array([0, 1] * 5, dtype=np.int64)
    result = apply_label_flip(X, y, poison_rate=0.3, seed=42)
    assert int((result.y != y).sum()) == 3
    assert result.manifest["poisoned_rows"] == 3


def test_attack_to_normal_only_changes_attacks() -> None:
    X = np.zeros((10, 2), dtype=np.float32)
    y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    result = apply_attack_to_normal(X, y, poison_rate=0.5, seed=7)
    changed = np.flatnonzero(result.y != y)
    assert changed.size == 3
    assert np.all(y[changed] == 1)
    assert np.all(result.y[changed] == 0)


def test_feature_noise_does_not_change_y() -> None:
    X = np.ones((12, 3), dtype=np.float32)
    y = np.array([0, 1] * 6, dtype=np.int64)
    result = apply_feature_noise(X, y, poison_rate=0.5, seed=123, std=0.1)
    assert np.array_equal(result.y, y)
    assert not np.array_equal(result.X, X)
    assert result.manifest["labels_changed"] == 0
