from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robustness.metrics import binary_metrics, robustness_score


def test_binary_metrics_counts_and_rates() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    metrics = binary_metrics(y_true, y_pred)
    assert metrics["TP"] == 2
    assert metrics["TN"] == 1
    assert metrics["FP"] == 1
    assert metrics["FN"] == 1
    assert metrics["FPR"] == 0.5
    assert metrics["FNR"] == 1 / 3


def test_robustness_score_penalizes_fpr() -> None:
    strong = {"macro_f1": 0.9, "attack_recall": 0.9, "FPR": 0.05}
    weak = {"macro_f1": 0.9, "attack_recall": 0.9, "FPR": 0.5}
    assert robustness_score(strong) > robustness_score(weak)
