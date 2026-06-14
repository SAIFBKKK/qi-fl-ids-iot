from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedtn_mps.config import load_config
from fedtn_mps.verify_setup import verify_setup


CONFIG = Path("experiments/qi-fl-ids-iot-final/configs/fedtn_mps_l1.yaml")


def test_verify_setup_accepts_with_checkpoint_warnings() -> None:
    summary = verify_setup(load_config(CONFIG), write_outputs=False)
    assert summary["accepted"] is True
    assert summary["criteria"]["full_fl_not_auto_launched"] is True
    assert len(summary["estimates"]) == 4
    assert any("checkpoint_not_available" in warning for warning in summary["warnings"])
