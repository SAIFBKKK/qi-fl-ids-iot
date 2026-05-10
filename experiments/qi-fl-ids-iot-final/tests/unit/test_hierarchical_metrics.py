from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fl_hierarchical.metrics import confusion_matrix, multiclass_metrics, one_vs_rest_rows  # noqa: E402


def test_confusion_matrix_shape() -> None:
    matrix = confusion_matrix(np.array([0, 1, 2]), np.array([0, 2, 2]), 3)
    assert matrix.shape == (3, 3)
    assert matrix.sum() == 3


def test_multiclass_metrics_bounds() -> None:
    metrics = multiclass_metrics(
        np.array([0, 1, 1, 2]),
        np.array([0, 1, 2, 2]),
        ["a", "b", "c"],
    )
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["weighted_f1"] <= 1.0


def test_one_vs_rest_rows_cover_all_classes() -> None:
    metrics = multiclass_metrics(
        np.array([0, 1, 2]),
        np.array([0, 1, 2]),
        ["a", "b", "c"],
    )
    rows = one_vs_rest_rows(metrics)
    assert len(rows) == 3
    assert {row["class_name"] for row in rows} == {"a", "b", "c"}
