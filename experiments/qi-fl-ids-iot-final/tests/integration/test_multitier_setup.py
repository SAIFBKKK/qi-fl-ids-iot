from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multitier_heterofl.config import load_config, tier_mapping_for_k  # noqa: E402
from multitier_heterofl.data import load_scenario, task_spec  # noqa: E402
from multitier_heterofl.verify_setup import verify_setup  # noqa: E402

CONFIG_PATH = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "configs" / "multitier_heterofl.yaml"
RUN_SCRIPT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src" / "scripts" / "07_run_multitier_heterofl.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_loads() -> None:
    config = load_config(CONFIG_PATH)
    assert config["tasks"]["run_l3"] is False


def test_tier_mapping_k3() -> None:
    config = load_config(CONFIG_PATH)
    assert tier_mapping_for_k(config, 3) == {"client_1": "weak", "client_2": "medium", "client_3": "powerful"}


def test_l1_l2_scenarios_exist() -> None:
    config = load_config(CONFIG_PATH)
    assert load_scenario(config, REPO_ROOT, task="l1", alpha=0.5, clients=3).global_test_npz.exists()
    assert load_scenario(config, REPO_ROOT, task="l2", alpha=0.5, clients=3).global_test_npz.exists()


def test_task_specs() -> None:
    config = load_config(CONFIG_PATH)
    assert task_spec(config, REPO_ROOT, "l1").output_dim == 2
    assert task_spec(config, REPO_ROOT, "l2").output_dim == 8


def test_run_script_importable() -> None:
    assert hasattr(_load_script(RUN_SCRIPT, "p7_run_script"), "main")


def test_verify_setup_runs_without_training() -> None:
    summary = verify_setup(CONFIG_PATH, write_outputs=False)
    assert summary["accepted"] is True
    assert summary["checks"]["run_l3_false"] is True
