from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import flwr as fl

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fl_hierarchical.data import load_hierarchical_config, load_l2_index_scenario, load_task_spec  # noqa: E402
from fl_hierarchical.legacy_client import start_legacy_client  # noqa: E402
from fl_hierarchical.runtime import configured_address  # noqa: E402
from fl_hierarchical.strategy import build_initial_parameters  # noqa: E402
from fl_hierarchical.summary_schema import expected_figures, expected_run_artifacts  # noqa: E402
from fl_hierarchical.verify_setup import verify_hierarchical_setup  # noqa: E402

CONFIG_PATH = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "configs" / "hierarchical_flower.yaml"
SERVER_SCRIPT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src" / "scripts" / "06_start_hierarchical_flower_server.py"
CLIENT_SCRIPT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src" / "scripts" / "06_start_hierarchical_flower_client.py"


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_loads() -> None:
    config = load_hierarchical_config(CONFIG_PATH)
    assert config["flower"]["strategy"] == "FedAvg"


def test_flower_version_detected() -> None:
    assert fl.__version__


def test_l2_l3_task_specs() -> None:
    config = load_hierarchical_config(CONFIG_PATH)
    assert load_task_spec(config, REPO_ROOT, "l2").output_dim == 8
    assert load_task_spec(config, REPO_ROOT, "l3").output_dim == 33


def test_p3_l2_partitions_exist() -> None:
    config = load_hierarchical_config(CONFIG_PATH)
    scenario = load_l2_index_scenario(config, REPO_ROOT, alpha=0.5, clients=3)
    assert all(client.train_row_ids_npy.exists() and client.val_row_ids_npy.exists() for client in scenario.clients)


def test_global_test_holdout_exists() -> None:
    config = load_hierarchical_config(CONFIG_PATH)
    scenario = load_l2_index_scenario(config, REPO_ROOT, alpha=0.5, clients=3)
    assert scenario.global_test_npz.exists()


def test_no_test_used_by_clients() -> None:
    source = inspect.getsource(start_legacy_client)
    assert "l2_test_npz" not in source
    assert "test_scaled.npz" not in source


def test_manual_scripts_import() -> None:
    assert hasattr(_load_script_module(SERVER_SCRIPT, "p6_server_script"), "main")
    assert hasattr(_load_script_module(CLIENT_SCRIPT, "p6_client_script"), "main")


def test_initial_parameters_build() -> None:
    config = load_hierarchical_config(CONFIG_PATH)
    params = build_initial_parameters(config, load_task_spec(config, REPO_ROOT, "l2"))
    assert params.tensors


def test_verify_setup_runs_without_training() -> None:
    summary = verify_hierarchical_setup(CONFIG_PATH, write_outputs=False)
    assert summary["accepted"] is True
    assert summary["checks"]["global_test_holdout_protected"] is True


def test_expected_contract_lists_are_populated() -> None:
    assert "runs/{run_id}/artifacts/run_summary.json" in expected_run_artifacts()
    assert any(item.endswith("l2_family_confusion_matrix.png") for item in expected_figures())


def test_port_config_loaded() -> None:
    config = load_hierarchical_config(CONFIG_PATH)
    assert configured_address(config) == "127.0.0.1:8081"
