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

from fl_l1_flower.client_app import create_client_app  # noqa: E402
from fl_l1_flower.data import load_flower_config, load_scenario  # noqa: E402
from fl_l1_flower.legacy_client import start_legacy_client  # noqa: E402
from fl_l1_flower.legacy_server import build_legacy_strategy  # noqa: E402
from fl_l1_flower.runtime import configured_address, prepare_run_paths  # noqa: E402
from fl_l1_flower.server_app import create_server_app  # noqa: E402
from fl_l1_flower.strategy import build_initial_parameters  # noqa: E402
from fl_l1_flower.summary_schema import expected_figures, expected_run_artifacts  # noqa: E402
from fl_l1_flower.verify_flower_setup import verify_flower_setup  # noqa: E402

CONFIG_PATH = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "configs" / "fl_l1_flower.yaml"
SERVER_SCRIPT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src" / "scripts" / "05_2_start_flower_server.py"
CLIENT_SCRIPT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src" / "scripts" / "05_2_start_flower_client.py"


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flower_config_loads() -> None:
    config = load_flower_config(CONFIG_PATH)
    assert config["flower"]["strategy"] == "FedAvg"


def test_flower_version_detected() -> None:
    assert fl.__version__


def test_p3_partitions_exist() -> None:
    config = load_flower_config(CONFIG_PATH)
    scenario = load_scenario(config, REPO_ROOT)
    assert all(client.train_npz.exists() and client.val_npz.exists() for client in scenario.clients)


def test_global_test_holdout_exists() -> None:
    assert (
        REPO_ROOT
        / "experiments"
        / "qi-fl-ids-iot-final"
        / "outputs"
        / "preprocessed"
        / "l1_binary"
        / "test_scaled.npz"
    ).exists()


def test_clients_detected() -> None:
    config = load_flower_config(CONFIG_PATH)
    scenario = load_scenario(config, REPO_ROOT)
    assert [client.client_id for client in scenario.clients] == ["client_1", "client_2", "client_3"]


def test_server_client_modules_importable() -> None:
    config = load_flower_config(CONFIG_PATH)
    scenario = load_scenario(config, REPO_ROOT)
    initial_parameters = build_initial_parameters(config)
    assert initial_parameters.tensors
    client_app = create_client_app(
        config=config,
        scenario=scenario,
        logs_dir=REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "outputs" / "tmp_flower_test_logs",
        max_samples_per_client=10,
    )
    assert client_app is not None


def test_legacy_runtime_fallback_importable() -> None:
    assert build_legacy_strategy is not None


def test_no_test_used_by_clients() -> None:
    config = load_flower_config(CONFIG_PATH)
    scenario = load_scenario(config, REPO_ROOT)
    assert not any((client.train_npz.parent / "test_scaled.npz").exists() for client in scenario.clients)


def test_manual_server_script_imports() -> None:
    module = _load_script_module(SERVER_SCRIPT, "flower_manual_server_script")
    assert hasattr(module, "main")


def test_manual_client_script_imports() -> None:
    module = _load_script_module(CLIENT_SCRIPT, "flower_manual_client_script")
    assert hasattr(module, "main")


def test_client_does_not_load_global_test() -> None:
    source = inspect.getsource(start_legacy_client)
    assert "global_test_npz" not in source
    assert "test_scaled.npz" not in source


def test_run_id_log_dirs_created(tmp_path: Path) -> None:
    config = {"outputs": {"run_dir": "outputs/fl_l1_flower"}}
    paths = prepare_run_paths(
        config=config,
        repo_root=tmp_path,
        alpha=0.5,
        clients=3,
        run_id="run_20990101_010203",
    )
    assert paths.run_id == "run_20990101_010203"
    assert paths.logs_dir.exists()
    assert paths.latest_run_path.exists()


def test_port_config_loaded() -> None:
    config = load_flower_config(CONFIG_PATH)
    assert config["flower"]["address"] == "127.0.0.1:8080"


def test_min_clients_match_k() -> None:
    config = load_flower_config(CONFIG_PATH)
    k = int(config["scenario"]["clients"])
    assert int(config["flower"]["min_fit_clients"]) == k
    assert int(config["flower"]["min_evaluate_clients"]) == k
    assert int(config["flower"]["min_available_clients"]) == k


def test_flower_client_address_configured() -> None:
    config = load_flower_config(CONFIG_PATH)
    assert configured_address(config) == "127.0.0.1:8080"
    assert configured_address(config, "127.0.0.1:8099") == "127.0.0.1:8099"


def test_verify_flower_setup_runs_without_training() -> None:
    summary = verify_flower_setup(CONFIG_PATH, write_outputs=False)
    assert summary["accepted"] is True
    assert summary["checks"]["no_test_used_by_clients"] is True
    assert "artifacts_expected" in summary
    assert "figures_expected" in summary
    assert "criteria" in summary
    assert "errors" in summary


def test_output_contract_expected_lists_are_populated() -> None:
    assert "runs/{run_id}/artifacts/run_summary.json" in expected_run_artifacts()
    assert any(item.endswith("fl_l1_flower_macro_f1_by_round.png") for item in expected_figures())
