from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qifa.config import load_config
from qifa.data import load_mask_info, load_scenario


CONFIG_PATH = Path("experiments/qi-fl-ids-iot-final/configs/qifa_l1.yaml")


def test_qifa_runtime_protects_global_test_holdout() -> None:
    config = load_config(CONFIG_PATH)
    scenario = load_scenario(config, Path.cwd().resolve(), alpha=0.5, clients=3)
    assert not any((client.train_npz.parent / "test_scaled.npz").exists() for client in scenario.clients)


def test_qifa_optional_qga_mask_available() -> None:
    config = load_config(CONFIG_PATH)
    mask_info = load_mask_info(config, use_qga_mask=True)
    assert mask_info["selected_mask_id"] == "conservative_seed_42"
    assert mask_info["selected_features_count"] == 12
