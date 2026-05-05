from __future__ import annotations

import math

import numpy as np

from src.scripts.evaluate_per_class_metrics import compute_per_class_metrics


def test_per_class_compute_on_known_matrix():
    cm = np.asarray(
        [
            [5, 1, 0],
            [2, 3, 1],
            [0, 4, 6],
        ],
        dtype=int,
    )

    rows = compute_per_class_metrics(cm)
    class_1 = rows[1]

    assert class_1["TP"] == 3
    assert class_1["FP"] == 5
    assert class_1["FN"] == 3
    assert class_1["TN"] == 11


def test_per_class_precision_recall_f1():
    cm = np.asarray(
        [
            [5, 1, 0],
            [2, 3, 1],
            [0, 4, 6],
        ],
        dtype=int,
    )

    class_1 = compute_per_class_metrics(cm)[1]

    assert abs(class_1["precision"] - (3 / 8)) < 1e-9
    assert abs(class_1["recall"] - (3 / 6)) < 1e-9
    assert abs(class_1["f1_score"] - (3 / 7)) < 1e-9


def test_per_class_handles_zero_division():
    cm = np.asarray(
        [
            [1, 0, 1],
            [0, 2, 0],
            [0, 0, 0],
        ],
        dtype=int,
    )

    class_2 = compute_per_class_metrics(cm)[2]

    assert class_2["support"] == 0
    assert math.isnan(class_2["recall"])
    assert math.isnan(class_2["f1_score"])


def test_per_class_returns_34_rows():
    cm = np.eye(34, dtype=int)

    rows = compute_per_class_metrics(cm)

    assert len(rows) == 34
