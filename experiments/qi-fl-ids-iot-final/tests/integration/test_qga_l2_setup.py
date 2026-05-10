"""Integration checks for P8-b QGA L2 setup."""

from __future__ import annotations

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga_l2.config import load_config, repo_path


CONFIG = Path("experiments/qi-fl-ids-iot-final/configs/qga_l2_feature_selection.yaml")


def test_qga_l2_config_loads() -> None:
    config = load_config(CONFIG)
    assert "qga_l2_profiles" in config
    assert config["model"]["output_dim"] == 8


def test_l2_inputs_and_partitions_exist() -> None:
    config = load_config(CONFIG)
    assert repo_path(config, "inputs.l2_train_npz").exists()
    assert repo_path(config, "inputs.l2_val_npz").exists()
    assert repo_path(config, "inputs.l2_test_npz").exists()
    assert (repo_path(config, "inputs.l2_partitions_root") / "alpha_0.5" / "k3").exists()


def test_l2_global_test_not_partitioned_to_clients() -> None:
    config = load_config(CONFIG)
    scenario = repo_path(config, "inputs.l2_partitions_root") / "alpha_0.5" / "k3"
    assert not any((scenario / f"client_{index}" / "test_scaled.npz").exists() for index in range(1, 4))
