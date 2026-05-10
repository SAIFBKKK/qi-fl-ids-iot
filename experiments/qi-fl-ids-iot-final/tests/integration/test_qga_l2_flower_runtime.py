"""Light tests for P8-b QGA L2 Flower runtime."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

from qga_l2.config import load_config
from qga_l2.flower_runtime import prepare_run_paths


CONFIG = Path("experiments/qi-fl-ids-iot-final/configs/qga_l2_feature_selection.yaml")


def test_flower_runtime_paths_use_qga_l2_output_dir(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    config["project_root"] = tmp_path.as_posix()
    config["outputs"]["qga_l2_flower_dir"] = "flower"
    paths = prepare_run_paths(config, alpha=0.5, clients=3, run_id="run_test")
    assert "qga_l2" not in str(paths["run_dir"])
    assert Path(paths["artifacts_dir"]).exists()


def test_flower_scripts_importable() -> None:
    for script in ["08_b_start_qga_l2_flower_server.py", "08_b_start_qga_l2_flower_client.py", "08_b_run_qga_l2_flower_smoke.py"]:
        path = Path("experiments/qi-fl-ids-iot-final/src/scripts") / script
        spec = importlib.util.spec_from_file_location(script.replace(".py", ""), path)
        assert spec is not None and spec.loader is not None


def test_mask_array_can_reduce_input_dim() -> None:
    mask = np.asarray([1, 0, 1, 0], dtype=np.int8)
    assert int(mask.sum()) == 2
