"""Tests for P8-b QGA L2 fitness."""

from __future__ import annotations

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

from qga_l2.fitness_l2 import compute_l2_fitness, macro_metrics_from_confusion, multiclass_confusion


def test_l2_fitness_formula() -> None:
    metrics = {"macro_f1": 0.8, "macro_recall": 0.7, "macro_fpr": 0.2}
    weights = {"macro_f1_weight": 0.60, "macro_recall_weight": 0.25, "macro_fpr_penalty": 0.10, "feature_penalty": 0.05}
    expected = 0.60 * 0.8 + 0.25 * 0.7 - 0.10 * 0.2 - 0.05 * (14 / 28)
    assert abs(compute_l2_fitness(metrics, 14, 28, weights) - expected) < 1e-12


def test_multiclass_metrics_include_macro_fpr() -> None:
    y_true = np.asarray([0, 0, 1, 1, 2, 2])
    y_pred = np.asarray([0, 1, 1, 1, 2, 0])
    matrix = multiclass_confusion(y_true, y_pred, 3)
    metrics = macro_metrics_from_confusion(matrix)
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["macro_fpr"] <= 1.0


def test_multiclass_macro_fpr_one_vs_rest_formula() -> None:
    matrix = np.asarray(
        [
            [5, 1, 0],
            [2, 3, 1],
            [0, 2, 4],
        ],
        dtype=np.int64,
    )
    metrics = macro_metrics_from_confusion(matrix)
    total = float(matrix.sum())
    expected_fprs = []
    expected_recalls = []
    for index in range(3):
        tp = float(matrix[index, index])
        fn = float(matrix[index, :].sum() - tp)
        fp = float(matrix[:, index].sum() - tp)
        tn = total - tp - fn - fp
        expected_fprs.append(fp / (fp + tn))
        expected_recalls.append(tp / (tp + fn))
    assert abs(metrics["macro_fpr"] - float(np.mean(expected_fprs))) < 1e-12
    assert abs(metrics["macro_recall"] - float(np.mean(expected_recalls))) < 1e-12
