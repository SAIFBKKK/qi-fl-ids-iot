"""Fast tests for P8 QGA fitness."""

from __future__ import annotations

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga.fitness import compute_qga_fitness, mask_constraint_status


def test_qga_fitness_formula() -> None:
    metrics = {"macro_f1": 0.9, "recall_attack": 0.8}
    weights = {"alpha_macro_f1": 0.6, "beta_attack_recall": 0.3, "lambda_feature_penalty": 0.1}
    score = compute_qga_fitness(metrics, features_count=14, total_features=28, weights=weights)
    assert abs(score - (0.6 * 0.9 + 0.3 * 0.8 - 0.1 * 0.5)) < 1e-12


def test_mask_constraint_status() -> None:
    assert mask_constraint_status(features_count=8, min_features=8, max_features=24)["valid"]
    assert not mask_constraint_status(features_count=7, min_features=8, max_features=24)["valid"]
    assert not mask_constraint_status(features_count=25, min_features=8, max_features=24)["valid"]
