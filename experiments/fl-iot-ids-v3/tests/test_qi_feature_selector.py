from __future__ import annotations

import json

import numpy as np

from src.qi.qi_feature_selector import (
    QIFeatureSelectorConfig,
    repair_exact_k,
    run_qi_feature_selection,
    save_qi_feature_selection_artifacts,
    sample_mask,
    theta_to_probabilities,
)


def _synthetic_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(96, 28)).astype(np.float32)
    y = rng.integers(0, 4, size=96, dtype=np.int64)
    X[:, :5] += y[:, None] * 0.4
    return X, y, [f"f{idx}" for idx in range(28)]


def test_theta_probabilities_and_repair_keep_exact_k():
    rng = np.random.default_rng(1)
    theta = np.full(28, np.pi / 4.0)
    probabilities = theta_to_probabilities(theta)
    mask = sample_mask(theta, 15, rng)
    repaired = repair_exact_k(np.zeros(28, dtype=bool), 15, rng, probabilities)

    assert probabilities.shape == (28,)
    assert np.allclose(probabilities, 0.5)
    assert mask.sum() == 15
    assert repaired.sum() == 15


def test_qi_feature_selector_is_deterministic_in_smoke_mode():
    X, y, feature_names = _synthetic_data()
    cfg = QIFeatureSelectorConfig(
        n_features=28,
        k_features=15,
        n_generations=2,
        pop_size=3,
        epochs=1,
        max_samples_per_class=4,
        seed=7,
        mode="smoke",
    )

    first = run_qi_feature_selection(X, y, feature_names, config=cfg, num_classes=34)
    second = run_qi_feature_selection(X, y, feature_names, config=cfg, num_classes=34)

    assert first.selected_indices == second.selected_indices
    assert len(first.selected_indices) == 15
    assert first.n_features == 28


def test_qi_feature_selector_saves_expected_artifacts(tmp_path):
    X, y, feature_names = _synthetic_data()
    result = run_qi_feature_selection(
        X,
        y,
        feature_names,
        config=QIFeatureSelectorConfig(
            n_features=28,
            k_features=10,
            n_generations=1,
            pop_size=2,
            epochs=1,
            max_samples_per_class=3,
            mode="smoke",
        ),
        num_classes=34,
    )

    paths = save_qi_feature_selection_artifacts(
        result,
        output_dir=tmp_path,
        scenario="synthetic",
    )

    payload = json.loads(paths["selected_features"].read_text(encoding="utf-8"))
    mask = np.load(paths["feature_mask"])

    assert payload["n_features"] == 28
    assert payload["k_features"] == 10
    assert len(payload["selected_indices"]) == 10
    assert mask.sum() == 10
    assert paths["selection_report"].exists()
