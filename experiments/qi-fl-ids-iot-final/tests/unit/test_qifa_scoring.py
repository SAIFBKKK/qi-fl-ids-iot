from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qifa.scoring import compute_client_score, normalize_scores_to_theta


def test_qifa_scores_calculated_correctly() -> None:
    weights = {
        "macro_f1_weight": 0.5,
        "attack_recall_weight": 0.25,
        "fpr_penalty": 0.1,
        "loss_penalty": 0.05,
        "drift_penalty": 0.1,
    }
    metrics = {
        "local_macro_f1": 0.8,
        "local_attack_recall": 0.9,
        "local_fpr": 0.2,
        "local_val_loss": 0.4,
        "drift": 0.3,
    }
    expected = 0.5 * 0.8 + 0.25 * 0.9 - 0.1 * 0.2 - 0.05 * 0.4 - 0.1 * 0.3
    assert abs(compute_client_score(metrics, weights) - expected) < 1e-12


def test_normalize_scores_stable_for_equal_values() -> None:
    theta = normalize_scores_to_theta([1.0, 1.0, 1.0])
    assert np.allclose(theta, np.pi / 4.0)
