from __future__ import annotations

import numpy as np
import pandas as pd

from src.scripts.generate_scenarios import (
    ROW_ID_COL,
    fit_train_only_scaler,
    split_raw_dataset,
)


def test_train_only_scaler_uses_train_statistics_only():
    train_df = pd.DataFrame(
        {
            ROW_ID_COL: [0, 1, 2, 3],
            "label_id": [0, 0, 1, 1],
            "feature_a": [0.0, 0.0, 2.0, 2.0],
            "feature_b": [10.0, 10.0, 20.0, 20.0],
        }
    )
    val_df = pd.DataFrame(
        {
            ROW_ID_COL: [4, 5],
            "label_id": [0, 1],
            "feature_a": [1000.0, 1000.0],
            "feature_b": [2000.0, 2000.0],
        }
    )

    scaler, feature_cols = fit_train_only_scaler(train_df)

    assert feature_cols == ["feature_a", "feature_b"]
    assert np.allclose(scaler.mean_, np.array([1.0, 15.0]))
    assert not np.allclose(
        scaler.mean_,
        pd.concat([train_df, val_df])[feature_cols].mean().to_numpy(),
    )


def test_raw_splits_are_disjoint_before_preprocessing():
    rows = []
    for label in range(2):
        for idx in range(20):
            rows.append({"label_id": label, "feature_a": float(idx), "feature_b": float(label)})
    df = pd.DataFrame(rows)

    splits = split_raw_dataset(df, seed=42)
    row_sets = {
        split: set(payload[ROW_ID_COL].astype(int).tolist())
        for split, payload in splits.items()
    }

    assert row_sets["train"].isdisjoint(row_sets["val"])
    assert row_sets["train"].isdisjoint(row_sets["test"])
    assert row_sets["val"].isdisjoint(row_sets["test"])
