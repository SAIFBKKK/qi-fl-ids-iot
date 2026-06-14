"""Light tests for P8 FedAvg+QGA true Flower runtime wiring."""

from __future__ import annotations

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga.config import load_config
from qga.flower_runtime import (
    _assert_full_run_uses_final_mask,
    build_qga_flower_config,
    load_mask_info,
    load_qga_flower_scenario,
)


CONFIG_PATH = Path("experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml")


def test_qga_flower_output_dir_configured() -> None:
    config = load_config(CONFIG_PATH)
    assert "qga_fedavg_flower_dir" in config["outputs"]


def test_calibrated_final_mask_is_default_for_flower_runtime() -> None:
    config = load_config(CONFIG_PATH)
    info = load_mask_info(config)
    assert int(info["mask"].sum()) == 12
    assert info["metadata"]["selected_mask_id"] == "conservative_seed_42"
    assert info["metadata"]["selected_mask_source"] == "final_selected_mask"
    assert info["metadata"]["calibration_decision_used"] is True


def test_qga_flower_config_uses_selected_input_dim() -> None:
    config = load_config(CONFIG_PATH)
    runtime = build_qga_flower_config(config, selected_count=13, alpha=0.5, clients=3, rounds=1)
    assert runtime["model"]["input_dim"] == 13
    assert runtime["flower"]["min_available_clients"] == 3


def test_full_flower_runtime_rejects_non_final_mask_path() -> None:
    config = load_config(CONFIG_PATH)
    old_mask_path = Path("experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/runs/run_20260508_104311/artifacts/feature_mask.json")
    if not old_mask_path.exists():
        old_mask_path = Path("not_final_feature_mask.json")
    try:
        _assert_full_run_uses_final_mask(config, mask_path=old_mask_path, mode="full")
    except ValueError as exc:
        assert "final_selected_mask" in str(exc)
    else:
        raise AssertionError("Expected full runtime to reject non-final explicit mask path")


def test_qga_flower_scenario_protects_test_holdout() -> None:
    config = load_config(CONFIG_PATH)
    scenario = load_qga_flower_scenario(config, Path.cwd().resolve(), alpha=0.5, clients=3)
    assert scenario.manifest["partition_test"] is False
    assert not any((client.train_npz.parent / "test_scaled.npz").exists() for client in scenario.clients)


def test_manual_flower_scripts_importable() -> None:
    import importlib.util

    for script in [
        "08_start_qga_fedavg_flower_server.py",
        "08_start_qga_fedavg_flower_client.py",
        "08_run_qga_fedavg_flower_smoke.py",
    ]:
        path = Path("experiments/qi-fl-ids-iot-final/src/scripts") / script
        spec = importlib.util.spec_from_file_location(script.replace(".py", ""), path)
        assert spec is not None
