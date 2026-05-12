from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robustness.config import load_config
from robustness.report_builder import build_reports


CONFIG = Path("experiments/qi-fl-ids-iot-final/configs/robustness_l1.yaml")


def test_report_builder_runs_with_or_without_runs() -> None:
    result = build_reports(load_config(CONFIG))
    assert Path(result["reports"][0]).exists()
    assert Path(result["reports"][1]).exists()
