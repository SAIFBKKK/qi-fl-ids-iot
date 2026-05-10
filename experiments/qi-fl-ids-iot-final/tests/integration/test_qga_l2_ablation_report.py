"""Tests for P8-b QGA L2 ablation script importability."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path("experiments/qi-fl-ids-iot-final/src/scripts/08_b_build_qga_l2_ablation_report.py")
    spec = importlib.util.spec_from_file_location("qga_l2_ablation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qga_l2_ablation_script_importable() -> None:
    _load_module()


def test_qga_l2_ablation_maps_l2_metric_aliases() -> None:
    module = _load_module()
    summary = {
        "accepted": True,
        "true_flower_runtime": True,
        "scenario": {"clients": 3, "rounds": 30},
        "qga": {"selected_features_count": 19},
        "dataset": {"input_dim_selected": 19},
        "test": {
            "metrics": {
                "macro_f1": 0.64,
                "weighted_f1": 0.77,
                "recall_macro": 0.71,
                "precision_macro": 0.66,
                "FPR_macro": 0.03,
                "accuracy": 0.75,
            }
        },
    }
    row = module._summary_to_row(summary, method="P8-b L2 FedAvg + QGA Flower", features_count=19, baseline_macro_f1=0.63)
    assert row["macro_recall"] == 0.71
    assert row["macro_precision"] == 0.66
    assert row["macro_fpr"] == 0.03
    assert abs(row["gap_macro_f1_vs_p6"] - 0.01) < 1e-12


def test_qga_l2_ablation_bandwidth_formula() -> None:
    module = _load_module()
    assert module.infer_l2_model_size_bytes(19) == 45344
    assert module.bandwidth_total_bytes(model_size_bytes=45344, clients=3, rounds=30) == 8161920


def test_qga_l2_ablation_warning_when_p6_missing(tmp_path) -> None:
    module = _load_module()
    config = {"project_root": str(tmp_path), "outputs": {"qga_l2_flower_dir": "missing", "reports_dir": "reports"}}
    summary, source = module._latest_p6_summary(config)
    assert summary is None
    assert source is None


def test_qga_l2_ablation_p8_row_not_empty() -> None:
    module = _load_module()
    summary = {
        "accepted": True,
        "criteria": {"true_flower_runtime": True},
        "scenario": {"clients": 3, "rounds": 30},
        "dataset": {"input_dim_selected": 19},
        "test": {"metrics": {"macro_f1": 0.646, "recall_macro": 0.717, "FPR_macro": 0.033}},
    }
    row = module._summary_to_row(summary, method="P8-b L2 FedAvg + QGA Flower", features_count=19)
    assert row["macro_f1"] == 0.646
    assert row["macro_recall"] == 0.717
    assert row["macro_fpr"] == 0.033
    assert row["model_size_bytes"] == 45344
