from __future__ import annotations

import json

import numpy as np

from src.qi.feature_selection import (
    QGAFeatureSelectionConfig,
    ensure_exact_k,
    run_qga_feature_selection,
    save_feature_selection_artifacts,
)


def _synthetic_data():
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(80, 28)).astype(np.float32)
    y_train = rng.integers(0, 3, size=80, dtype=np.int64)
    X_val = rng.normal(size=(40, 28)).astype(np.float32)
    y_val = rng.integers(0, 3, size=40, dtype=np.int64)
    X_train[:, :4] += y_train[:, None] * 0.5
    X_val[:, :4] += y_val[:, None] * 0.5
    return X_train, y_train, X_val, y_val, [f"f{idx}" for idx in range(28)]


def test_ensure_exact_k_repairs_masks():
    rng = np.random.default_rng(1)
    mask = np.zeros(28, dtype=bool)
    repaired = ensure_exact_k(mask, 16, rng)

    assert repaired.sum() == 16


def test_qga_feature_selection_returns_stable_k_features():
    X_train, y_train, X_val, y_val, feature_names = _synthetic_data()
    cfg = QGAFeatureSelectionConfig(
        k_features=16,
        population_size=8,
        generations=4,
        random_seed=7,
    )

    first = run_qga_feature_selection(
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names,
        config=cfg,
        smoke=True,
    )
    second = run_qga_feature_selection(
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names,
        config=cfg,
        smoke=True,
    )

    assert len(first.selected_indices) == 16
    assert first.selected_indices == second.selected_indices
    assert all(0 <= index < 28 for index in first.selected_indices)


def test_qga_feature_selection_saves_expected_artifacts(tmp_path):
    X_train, y_train, X_val, y_val, feature_names = _synthetic_data()
    result = run_qga_feature_selection(
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names,
        config=QGAFeatureSelectionConfig(k_features=8, population_size=6, generations=2),
        smoke=True,
    )

    paths = save_feature_selection_artifacts(
        result,
        output_dir=tmp_path,
        scenario="synthetic",
    )

    payload = json.loads(paths["selected_features"].read_text(encoding="utf-8"))
    mask = np.load(paths["feature_mask"])

    assert payload["k_features"] == 8
    assert len(payload["selected_indices"]) == 8
    assert mask.sum() == 8
    assert paths["selection_report"].exists()
