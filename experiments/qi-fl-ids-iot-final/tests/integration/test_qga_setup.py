"""Light integration tests for P8 QGA setup."""

from __future__ import annotations

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga.config import load_config, repo_path
from qga.feature_mask import load_feature_names
from qga.verify_setup import verify_qga_setup


CONFIG_PATH = Path("experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml")


def test_qga_config_loads() -> None:
    config = load_config(CONFIG_PATH)
    assert config["qga"]["weights"]["alpha_macro_f1"] == 0.6
    assert config["qga"]["min_features"] == 8


def test_qga_inputs_exist() -> None:
    config = load_config(CONFIG_PATH)
    for key in ["inputs.train_npz", "inputs.val_npz", "inputs.test_npz", "inputs.feature_names"]:
        assert repo_path(config, key).exists()


def test_feature_names_count_is_28() -> None:
    config = load_config(CONFIG_PATH)
    assert len(load_feature_names(repo_path(config, "inputs.feature_names"))) == 28


def test_verify_setup_accepts() -> None:
    config = load_config(CONFIG_PATH)
    summary = verify_qga_setup(config)
    assert summary["accepted"] is True
    assert summary["global_test_holdout"]["used_for_selection"] is False
