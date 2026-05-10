"""Tests for P8.1.5 QGA calibration profiles."""

from __future__ import annotations

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga.calibration import build_profile_params, calibration_seeds, get_qga_profiles
from qga.config import load_config
from qga.fitness import compute_qga_fitness


CONFIG_PATH = Path("experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml")


def test_qga_profiles_are_read() -> None:
    config = load_config(CONFIG_PATH)
    profiles = get_qga_profiles(config)
    assert set(profiles) == {"balanced_current", "conservative", "fpr_aware", "compression"}
    assert profiles["fpr_aware"]["fpr_penalty"] == 0.10


def test_calibration_seeds_are_configured() -> None:
    config = load_config(CONFIG_PATH)
    assert calibration_seeds(config) == [42, 123, 2026]


def test_profile_params_override_feature_bounds_and_seed() -> None:
    config = load_config(CONFIG_PATH)
    params = build_profile_params(config, profile_name="compression", seed=123, generations=2, population_size=4)
    assert params["seed"] == 123
    assert params["generations"] == 2
    assert params["population_size"] == 4
    assert params["min_features"] == 8
    assert params["max_features"] == 18
    assert params["weights"]["lambda_feature_penalty"] == 0.3


def test_fpr_penalty_is_applied_to_fitness() -> None:
    metrics = {"macro_f1": 0.9, "recall_attack": 0.8, "FPR": 0.2}
    score = compute_qga_fitness(
        metrics,
        features_count=14,
        total_features=28,
        weights={
            "alpha_macro_f1": 0.55,
            "beta_attack_recall": 0.25,
            "lambda_feature_penalty": 0.10,
            "fpr_penalty": 0.10,
        },
    )
    assert abs(score - (0.55 * 0.9 + 0.25 * 0.8 - 0.10 * 0.5 - 0.10 * 0.2)) < 1e-12
