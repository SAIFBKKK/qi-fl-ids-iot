from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qifa.config import load_config
from qifa.report_builder import build_qifa_ablation_report


CONFIG_PATH = Path("experiments/qi-fl-ids-iot-final/configs/qifa_l1.yaml")


def test_qifa_ablation_builder_reads_baselines() -> None:
    rows = build_qifa_ablation_report(load_config(CONFIG_PATH), Path.cwd().resolve())
    methods = {row.get("method") for row in rows}
    assert "P5 FedAvg baseline" in methods
    assert "P8 FedAvg + QGA Flower" in methods


def test_qifa_ablation_script_importable() -> None:
    path = Path("experiments/qi-fl-ids-iot-final/src/scripts/09_build_qifa_ablation_report.py")
    spec = importlib.util.spec_from_file_location("qifa_ablation_script", path)
    assert spec is not None and spec.loader is not None
