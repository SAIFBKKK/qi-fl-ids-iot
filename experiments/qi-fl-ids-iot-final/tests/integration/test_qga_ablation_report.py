"""Integration tests for P8 QGA ablation source selection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


SCRIPT_PATH = Path("experiments/qi-fl-ids-iot-final/src/scripts/08_build_qga_ablation_report.py")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("build_qga_ablation_report", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(*, run_id: str, features: int, calibrated: bool, mode: str = "full") -> dict:
    mask_id = "conservative_seed_42" if calibrated else "balanced_current_seed_42"
    return {
        "accepted": True,
        "run_id": run_id,
        "mode": mode,
        "runtime": "manual",
        "true_flower_runtime": True,
        "selected_mask_id": mask_id,
        "selected_mask_source": "final_selected_mask" if calibrated else "latest_qga_run",
        "calibration_decision_used": calibrated,
        "scenario": {"alpha": 0.5, "clients": 3, "rounds": 30},
        "training": {"rounds_completed": 30, "rounds_configured": 30},
        "dataset": {
            "input_dim_selected": features,
            "test_sent_to_clients": False,
        },
        "qga": {
            "selected_mask_id": mask_id,
            "selected_mask_source": "final_selected_mask" if calibrated else "latest_qga_run",
            "calibration_decision_used": calibrated,
            "selected_features_count": features,
        },
        "criteria": {
            "true_flower_runtime": True,
            "test_sent_to_clients_false": True,
        },
        "test": {"metrics": {"macro_f1": 0.94, "recall_attack": 0.95, "FPR": 0.05}},
        "communication": {"model_size_bytes": 40_200, "communication_cumulative_bytes": 7_236_000},
    }


def _write_run(root: Path, run_id: str, summary: dict) -> None:
    path = root / "alpha_0.5" / "k3" / "runs" / run_id / "artifacts" / "run_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary), encoding="utf-8")


def test_ablation_prefers_calibrated_flower_run_and_ignores_old_mask(tmp_path: Path) -> None:
    module = _load_script_module()
    flower_root = tmp_path / "flower"
    _write_run(flower_root, "run_20260508_111701", _summary(run_id="run_20260508_111701", features=9, calibrated=False))
    _write_run(flower_root, "run_20260508_160000", _summary(run_id="run_20260508_160000", features=12, calibrated=True))
    config = {
        "project_root": tmp_path.as_posix(),
        "outputs": {
            "qga_fedavg_flower_dir": "flower",
            "qga_fedavg_dir": "inprocess",
        },
    }

    summary, warnings = module.find_latest_valid_qga_flower_run(config, rounds=30)

    assert summary["run_id"] == "run_20260508_160000"
    assert summary["calibration_decision_used"] is True
    assert summary["selected_mask_source"] == "final_selected_mask"
    assert summary["qga"]["selected_features_count"] == 12
    assert any("non-calibrated" in warning for warning in warnings)


def test_ablation_rejects_only_old_nine_feature_flower_runs(tmp_path: Path) -> None:
    module = _load_script_module()
    flower_root = tmp_path / "flower"
    _write_run(flower_root, "run_20260508_152046", _summary(run_id="run_20260508_152046", features=9, calibrated=False))
    config = {
        "project_root": tmp_path.as_posix(),
        "outputs": {
            "qga_fedavg_flower_dir": "flower",
            "qga_fedavg_dir": "inprocess",
        },
    }

    try:
        module.find_latest_valid_qga_flower_run(config, rounds=30)
    except RuntimeError as exc:
        assert "calibration_decision_used=true" in str(exc)
    else:
        raise AssertionError("Expected old non-calibrated Flower run to be rejected")
