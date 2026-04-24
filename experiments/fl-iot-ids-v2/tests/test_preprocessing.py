import numpy as np
import pandas as pd

from src.data.datasets.flat_dataset import FlatTensorDataset
from src.scripts.prepare_partitions import (
    RARE_CLASSES,
    RARE_EXPERT_NODE,
    ROW_ID_COL,
    build_rare_expert,
    validate_disjoint_partitions,
)
import src.scripts.fit_global_scaler as fit_scaler_module
from src.data.preprocessing.preprocessor import LocalNodePreprocessor


def test_flat_tensor_dataset_shapes():
    x = np.random.randn(10, 33).astype(np.float32)
    y = np.random.randint(0, 34, size=(10,), dtype=np.int64)
    ds = FlatTensorDataset(x, y)
    assert len(ds) == 10


def test_rare_expert_partitions_are_row_disjoint_and_exclusive():
    rows = []
    row_id = 0
    for label in [1, 4, 5, *sorted(RARE_CLASSES)]:
        for idx in range(12):
            rows.append(
                {
                    ROW_ID_COL: row_id,
                    "label_id": label,
                    "feature_a": float(idx),
                    "feature_b": float(label),
                }
            )
            row_id += 1
    df = pd.DataFrame(rows)

    partitions = build_rare_expert(df, label_col="label_id", seed=42)
    proof = validate_disjoint_partitions(partitions)

    assert proof["disjoint"] is True
    for node_id, part_df in partitions.items():
        rare_present = set(part_df["label_id"].unique()) & RARE_CLASSES
        if node_id == RARE_EXPERT_NODE:
            assert rare_present == RARE_CLASSES
        else:
            assert rare_present == set()


def test_v2_preprocessor_uses_global_scaler_artifact(tmp_path, monkeypatch):
    scenario = "normal_noniid"
    raw_node = tmp_path / "raw" / scenario / "node1"
    raw_node.mkdir(parents=True)
    train_df = pd.DataFrame(
        {
            ROW_ID_COL: [0, 1, 2, 3],
            "label_id": [0, 0, 1, 1],
            "feature_a": [0.0, 1.0, 2.0, 3.0],
            "feature_b": [10.0, 11.0, 12.0, 13.0],
        }
    )
    train_df.to_csv(raw_node / "train.csv", index=False)

    monkeypatch.setattr(fit_scaler_module, "DATA_DIR", tmp_path)
    artifacts_dir = tmp_path / "artifacts" / "shared"
    fit_scaler_module.fit_global_scaler_for_scenario(scenario, artifacts_dir=artifacts_dir)

    preprocessor = LocalNodePreprocessor(artifacts_dir=artifacts_dir, scenario=scenario)
    preprocessor.load_artifacts()
    X, y, feature_names = preprocessor.transform_dataframe(train_df)

    assert X.shape == (4, 2)
    assert y.tolist() == [0, 0, 1, 1]
    assert feature_names == ["feature_a", "feature_b"]
