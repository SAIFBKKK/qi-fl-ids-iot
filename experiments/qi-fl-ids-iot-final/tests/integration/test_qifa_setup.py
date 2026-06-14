from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qifa.config import load_config
from qifa.verify_setup import verify_qifa_setup


CONFIG_PATH = Path("experiments/qi-fl-ids-iot-final/configs/qifa_l1.yaml")


def test_qifa_config_loads() -> None:
    config = load_config(CONFIG_PATH)
    assert config["default_variant"] == "hybrid"


def test_qifa_verify_setup_accepted() -> None:
    summary = verify_qifa_setup(CONFIG_PATH, write_outputs=False)
    assert summary["accepted"] is True
    assert summary["checks"]["qga_final_mask_ready"] is True


def test_qifa_server_client_scripts_importable() -> None:
    for script in [
        "09_verify_qifa_setup.py",
        "09_run_qifa_flower_smoke.py",
        "09_start_qifa_flower_server.py",
        "09_start_qifa_flower_client.py",
        "09_build_qifa_ablation_report.py",
    ]:
        path = Path("experiments/qi-fl-ids-iot-final/src/scripts") / script
        spec = importlib.util.spec_from_file_location(script.replace(".py", ""), path)
        assert spec is not None and spec.loader is not None
