import json

import numpy as np
import pytest

from src.common.preflight import validate_required_artifacts
from src.common.runtime import configure_runtime_artifacts


def _base_config(scenario: str = "rare_expert") -> dict:
    return {
        "scenario": {"name": scenario, "num_clients": 3},
        "imbalance": {"name": "class_weights"},
    }


def test_validate_required_artifacts_rejects_legacy_manifest(tmp_path):
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    manifest_path = data_dir / "splits" / "rare_expert_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"scenario": "rare_expert"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Legacy manifest detected"):
        validate_required_artifacts(
            _base_config(),
            ["node1", "node2", "node3"],
            data_dir=data_dir,
            artifacts_dir=artifacts_dir,
        )


def test_validate_required_artifacts_accepts_split_manifest_and_npz(tmp_path):
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    manifest_path = data_dir / "splits" / "rare_expert_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"scenario": "rare_expert", "splits": {"train": {}, "val": {}, "test": {}}}),
        encoding="utf-8",
    )

    for node_id in ("node1", "node2", "node3"):
        node_dir = data_dir / "processed" / "rare_expert" / node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            np.savez(node_dir / f"{split}_preprocessed.npz", X=np.zeros((1, 1)), y=np.zeros((1,)))

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "class_weights_rare_expert.pkl").write_bytes(b"ok")

    validate_required_artifacts(
        _base_config(),
        ["node1", "node2", "node3"],
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )


def test_configure_runtime_artifacts_assigns_isolated_scaffold_state_dir():
    experiment = {
        "name": "exp_scaffold_test",
        "fl_strategy": "scaffold",
        "data_scenario": "rare_expert",
        "imbalance_strategy": "class_weights",
    }
    config: dict = {}

    run_name = configure_runtime_artifacts(experiment, config)

    assert config["runtime"]["run_name"] == run_name
    assert str(config["runtime"]["scaffold_state_dir"]).endswith(run_name)
