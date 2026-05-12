from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robustness.config import load_config
from robustness.verify_setup import verify_setup


CONFIG = Path("experiments/qi-fl-ids-iot-final/configs/robustness_l1.yaml")


def test_verify_setup_accepted() -> None:
    summary = verify_setup(load_config(CONFIG), write_outputs=False)
    assert summary["accepted"] is True
    assert summary["criteria"]["global_test_holdout_protected"] is True


def test_config_declares_required_attacks() -> None:
    config = load_config(CONFIG)
    assert set(config["attack_types"]) == {"clean", "label_flip", "attack_to_normal", "feature_noise"}
    assert set(config["methods"]) == {"fedavg", "fedavg_qga", "qifa", "qifa_qga"}
