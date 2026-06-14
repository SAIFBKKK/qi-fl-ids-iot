"""Fast tests for QGA feature-mask helpers."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga.feature_mask import apply_feature_mask, mask_payload, selected_indices


def test_selected_indices() -> None:
    mask = np.array([1, 0, 1, 0], dtype=np.int8)
    assert selected_indices(mask) == [0, 2]


def test_apply_feature_mask_reduces_columns() -> None:
    X = np.arange(12).reshape(3, 4)
    mask = np.array([1, 0, 1, 0], dtype=np.int8)
    reduced = apply_feature_mask(X, mask)
    assert reduced.shape == (3, 2)
    assert reduced[:, 0].tolist() == [0, 4, 8]


def test_mask_payload_contains_selected_names() -> None:
    payload = mask_payload(
        np.array([1, 0, 1], dtype=np.int8),
        ["a", "b", "c"],
        run_id="run_test",
        method="test",
    )
    assert payload["selected_indices"] == [0, 2]
    assert payload["selected_features"] == ["a", "c"]
