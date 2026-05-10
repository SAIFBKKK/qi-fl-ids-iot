from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fl_l1.fedavg_server import resolve_sample_limit_for_mode  # noqa: E402
from fl_l1.round_logger import format_round_console_line  # noqa: E402
from fl_l1.scenario_loader import load_config, load_l1_scenario  # noqa: E402
from fl_l1.verify_setup import verify_setup  # noqa: E402
from models.l1_mlp import CentralizedL1MLP  # noqa: E402

CONFIG_PATH = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "configs" / "fl_l1_fedavg.yaml"


def test_config_loads() -> None:
    config = load_config(CONFIG_PATH)
    assert config["federated"]["strategy"] == "FedAvg"


def test_p3_l1_partitions_exist() -> None:
    root = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final" / "outputs" / "partitions" / "l1_binary"
    assert (root / "alpha_0.5" / "k3" / "manifest.json").exists()


def test_p4_metrics_exist() -> None:
    assert (
        REPO_ROOT
        / "experiments"
        / "qi-fl-ids-iot-final"
        / "outputs"
        / "centralized_l1"
        / "artifacts"
        / "metrics_test.json"
    ).exists()


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


def test_scenario_loader_alpha_0_5_k3() -> None:
    config = load_config(CONFIG_PATH)
    scenario = load_l1_scenario(config, REPO_ROOT, alpha=0.5, num_clients=3)
    assert scenario.alpha == 0.5
    assert scenario.num_clients == 3
    assert len(scenario.clients) == 3
    assert all(client.train_npz.exists() and client.val_npz.exists() for client in scenario.clients)


def test_model_architecture_matches_p4() -> None:
    config = load_config(CONFIG_PATH)
    model_cfg = config["model"]
    model = CentralizedL1MLP(
        input_dim=model_cfg["input_dim"],
        hidden_layers=model_cfg["hidden_layers"],
        output_dim=model_cfg["output_dim"],
        dropout=model_cfg["dropout"],
        activation=model_cfg["activation"],
    )
    assert model.count_parameters() == 12_098


def test_verify_setup_runs_without_training() -> None:
    summary = verify_setup(CONFIG_PATH, write_outputs=False)
    assert summary["accepted"] is True
    assert summary["checks"]["verify_runs_without_training"] is True


def test_full_mode_does_not_use_smoke_sampling() -> None:
    assert (
        resolve_sample_limit_for_mode(
            mode="full",
            requested_max_samples=1000,
            default_smoke_max_samples=1000,
        )
        is None
    )
    assert (
        resolve_sample_limit_for_mode(
            mode="grid",
            requested_max_samples=1000,
            default_smoke_max_samples=1000,
        )
        is None
    )
    assert (
        resolve_sample_limit_for_mode(
            mode="smoke",
            requested_max_samples=None,
            default_smoke_max_samples=1000,
        )
        == 1000
    )


def test_round_console_log_format() -> None:
    line = format_round_console_line(
        {
            "round": 3,
            "alpha": 0.5,
            "num_clients": 3,
            "train_loss_mean": 0.12345,
            "val_loss_mean": 0.23456,
            "macro_f1": 0.9407,
            "attack_recall": 0.9474,
            "FPR": 0.0663,
            "TP": 46897,
            "TN": 42018,
            "FP": 2982,
            "FN": 2603,
            "round_time_sec": 1.25,
            "model_size_bytes": 48392,
            "communication_upload_bytes": 145176,
            "communication_download_bytes": 145176,
            "communication_total_bytes": 290352,
            "communication_cumulative_bytes": 8710560,
        },
        current_round=3,
        total_rounds=30,
    )
    assert line.startswith("[Round 03/30]")
    assert "macro_f1=0.9407" in line
    assert "attack_recall=0.9474" in line
    assert "FPR=0.0663" in line
    assert "bytes=290352" in line
    assert "cum=8710560" in line


def test_run_console_log_created_after_smoke() -> None:
    log_path = (
        REPO_ROOT
        / "experiments"
        / "qi-fl-ids-iot-final"
        / "outputs"
        / "fl_l1_fedavg"
        / "alpha_0.1"
        / "k3"
        / "logs"
        / "run_console.log"
    )
    if not log_path.exists():
        pytest.skip("run smoke alpha=0.1 k3 before asserting run_console.log")
    text = log_path.read_text(encoding="utf-8")
    assert "Starting FedAvg L1 server" in text
    assert "[Round 01/01]" in text
