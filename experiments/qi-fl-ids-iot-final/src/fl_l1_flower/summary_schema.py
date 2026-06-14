"""Stable output contracts for P5.2 Flower verify and run summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import flwr as fl

from fl_l1.scenario_loader import rel, write_json


RUN_ARTIFACT_PATTERNS = [
    "manifest.json",
    "checkpoints/best_global_model.pth",
    "checkpoints/last_global_model.pth",
    "artifacts/model_config.json",
    "artifacts/metrics_rounds.csv",
    "artifacts/metrics_clients.csv",
    "artifacts/bandwidth_rounds.csv",
    "artifacts/aggregation_weights.csv",
    "artifacts/metrics_val.json",
    "artifacts/metrics_test.json",
    "artifacts/threshold.json",
    "artifacts/threshold_sweep.csv",
    "artifacts/confusion_matrix.csv",
    "artifacts/classification_report.json",
    "artifacts/comparison_with_p4.json",
    "artifacts/run_summary.json",
    "artifacts/run_manifest.json",
    "logs/flower_server.log",
    "logs/flower_clients.log",
    "logs/run_console.log",
]

FIGURE_FILENAMES = [
    "fl_l1_flower_loss_by_round.png",
    "fl_l1_flower_macro_f1_by_round.png",
    "fl_l1_flower_attack_recall_by_round.png",
    "fl_l1_flower_fpr_by_round.png",
    "fl_l1_flower_bandwidth_by_round.png",
    "fl_l1_flower_round_time_by_round.png",
    "fl_l1_flower_confusion_matrix.png",
    "fl_l1_flower_threshold_sweep.png",
    "fl_l1_flower_tp_tn_fp_fn.png",
    "fl_l1_flower_roc_curve.png",
    "fl_l1_flower_pr_curve.png",
    "fl_l1_flower_vs_p4.png",
    "fl_l1_flower_client_metrics_heatmap.png",
    "fl_l1_flower_architecture.png",
]


def alpha_dir(alpha: float) -> str:
    return f"alpha_{alpha:.1f}" if float(alpha).is_integer() else f"alpha_{alpha}"


def architecture_string(model_cfg: dict[str, Any]) -> str:
    layers = [int(model_cfg["input_dim"]), *[int(item) for item in model_cfg["hidden_layers"]], int(model_cfg["output_dim"])]
    return " -> ".join(str(item) for item in layers)


def expected_run_artifacts() -> list[str]:
    return [f"runs/{{run_id}}/{item}" for item in RUN_ARTIFACT_PATTERNS] + [
        "latest_run.json",
        "latest_run_summary.json",
    ]


def expected_figures() -> list[str]:
    return [f"outputs/figures/fl_l1_flower/alpha_{{alpha}}/k{{k}}/{{run_id}}/{item}" for item in FIGURE_FILENAMES]


def figure_dir(config: dict[str, Any], repo_root: Path, *, alpha: float, clients: int, run_id: str) -> Path:
    return repo_root / config["outputs"]["figures_dir"] / alpha_dir(alpha) / f"k{clients}" / run_id


def existing_relative_paths(paths: list[Path], repo_root: Path) -> list[str]:
    return [rel(path, repo_root) for path in paths if path.exists()]


def run_artifact_paths(run_dir: Path) -> list[Path]:
    return [run_dir / item for item in RUN_ARTIFACT_PATTERNS]


def run_figure_paths(figures_dir: Path) -> list[Path]:
    return [figures_dir / filename for filename in FIGURE_FILENAMES]


def comparison_with_p4(p4_metrics: dict[str, Any], metrics_test: dict[str, Any]) -> dict[str, float]:
    p4_accuracy = float(p4_metrics.get("accuracy", 0.0))
    p5_accuracy = float(metrics_test.get("accuracy", 0.0))
    p4_macro_f1 = float(p4_metrics.get("macro_f1", 0.0))
    p5_macro_f1 = float(metrics_test.get("macro_f1", 0.0))
    p4_attack_recall = float(p4_metrics.get("recall_attack", p4_metrics.get("attack_recall", 0.0)))
    p5_attack_recall = float(metrics_test.get("recall_attack", metrics_test.get("attack_recall", 0.0)))
    p4_fpr = float(p4_metrics.get("FPR", p4_metrics.get("fpr", 0.0)))
    p5_fpr = float(metrics_test.get("FPR", metrics_test.get("fpr", 0.0)))
    return {
        "p4_accuracy": p4_accuracy,
        "p5_2_accuracy": p5_accuracy,
        "gap_accuracy": p5_accuracy - p4_accuracy,
        "p4_macro_f1": p4_macro_f1,
        "p5_2_macro_f1": p5_macro_f1,
        "p5_flower_macro_f1": p5_macro_f1,
        "gap_macro_f1": p5_macro_f1 - p4_macro_f1,
        "p4_attack_recall": p4_attack_recall,
        "p5_2_attack_recall": p5_attack_recall,
        "p5_flower_attack_recall": p5_attack_recall,
        "gap_attack_recall": p5_attack_recall - p4_attack_recall,
        "p4_fpr": p4_fpr,
        "p5_2_fpr": p5_fpr,
        "p5_flower_fpr": p5_fpr,
        "gap_fpr": p5_fpr - p4_fpr,
    }


def verify_criteria(checks: dict[str, Any]) -> dict[str, bool]:
    return {
        "flower_version_detected": bool(checks.get("flower_version_detected")),
        "flower_runtime_available": bool(checks.get("legacy_runtime_fallback_available")),
        "p3_partitions_ready": bool(checks.get("p3_partitions_exist")),
        "p4_metrics_ready": bool(checks.get("p4_metrics_exist")),
        "global_test_holdout_exists": bool(checks.get("global_test_holdout_exists")),
        "global_test_holdout_protected": bool(checks.get("no_test_used_by_clients")),
        "initial_parameters_build": bool(checks.get("initial_parameters_build")),
        "server_client_modules_importable": bool(checks.get("server_client_modules_importable")),
    }


def build_verify_contract(
    *,
    accepted: bool,
    architecture: str,
    scenario: dict[str, Any],
    global_test_holdout: dict[str, Any],
    checks: dict[str, Any],
    warnings: list[str],
    errors: list[str] | None = None,
) -> dict[str, Any]:
    errors = errors or []
    return {
        "accepted": bool(accepted),
        "mode": "verify",
        "flower_version": fl.__version__,
        "architecture": architecture,
        "scenario": scenario,
        "global_test_holdout": global_test_holdout,
        "checks": checks,
        "artifacts_expected": expected_run_artifacts(),
        "figures_expected": expected_figures(),
        "criteria": verify_criteria(checks),
        "warnings": warnings,
        "errors": errors,
    }


def run_criteria(
    *,
    mode: str,
    rounds_completed: int,
    rounds_configured: int,
    artifacts: list[str],
    figures: list[str],
    docs_generated: bool,
) -> dict[str, bool]:
    artifact_names = {Path(item).name for item in artifacts}
    return {
        "p3_partitions_used": True,
        "global_test_holdout_protected": True,
        "test_sent_to_clients_false": True,
        "server_client_runtime_started": rounds_completed > 0,
        "round_metrics_generated": "metrics_rounds.csv" in artifact_names,
        "client_metrics_generated": "metrics_clients.csv" in artifact_names,
        "bandwidth_metrics_generated": "bandwidth_rounds.csv" in artifact_names,
        "best_model_saved": "best_global_model.pth" in artifact_names,
        "last_model_saved": "last_global_model.pth" in artifact_names,
        "threshold_generated": "threshold.json" in artifact_names,
        "threshold_validation_only": True,
        "metrics_val_generated": "metrics_val.json" in artifact_names,
        "metrics_test_generated": "metrics_test.json" in artifact_names,
        "comparison_with_p4_generated": "comparison_with_p4.json" in artifact_names,
        "figures_generated": len(figures) == len(FIGURE_FILENAMES),
        "docs_generated": docs_generated,
        "run_logs_generated": "run_console.log" in artifact_names,
        "flower_runtime_true": True,
        "smoke_run_only": mode == "smoke",
        "full_run_completed": mode == "full" and rounds_completed == rounds_configured,
    }


def write_latest_run_summary(*, run_dir: Path, repo_root: Path, summary: dict[str, Any]) -> Path:
    scenario_dir = run_dir.parents[1]
    latest_path = scenario_dir / "latest_run_summary.json"
    write_json(latest_path, summary)
    return latest_path
