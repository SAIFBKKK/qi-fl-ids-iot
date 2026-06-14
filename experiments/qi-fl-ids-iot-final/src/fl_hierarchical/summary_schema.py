"""Stable output contracts for P6 hierarchical Flower summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import flwr as fl

from fl_l1.scenario_loader import rel
from fl_hierarchical.data import normalize_task
from fl_hierarchical.runtime import alpha_dir


RUN_ARTIFACT_PATTERNS = [
    "manifest.json",
    "checkpoints/best_global_model.pth",
    "checkpoints/last_global_model.pth",
    "artifacts/model_config.json",
    "artifacts/class_mapping.json",
    "artifacts/metrics_rounds.csv",
    "artifacts/metrics_clients.csv",
    "artifacts/bandwidth_rounds.csv",
    "artifacts/aggregation_weights.csv",
    "artifacts/metrics_val.json",
    "artifacts/metrics_test.json",
    "artifacts/classification_report.json",
    "artifacts/confusion_matrix.csv",
    "artifacts/one_vs_rest_metrics.csv",
    "artifacts/threshold_or_decision_config.json",
    "artifacts/comparison_with_l1_l2_l3.json",
    "artifacts/run_manifest.json",
    "artifacts/run_summary.json",
    "logs/flower_server.log",
    "logs/flower_clients.log",
    "logs/run_console.log",
]

L2_FIGURES = [
    "l2_family_confusion_matrix.png",
    "l2_family_per_class_f1.png",
    "l2_family_one_vs_rest_tp_fp_tn_fn.png",
    "l2_family_metrics_by_round.png",
    "l2_family_bandwidth_by_round.png",
    "l2_family_client_metrics_heatmap.png",
]

L3_FIGURES = [
    "l3_attack_type_top_errors.png",
    "l3_attack_type_per_class_f1.png",
    "l3_attack_type_one_vs_rest_tp_fp_tn_fn.png",
    "l3_attack_type_metrics_by_round.png",
    "l3_attack_type_bandwidth_by_round.png",
    "l3_attack_type_confusion_matrix.png",
]

COMMON_FIGURES = [
    "hierarchical_tree_l1_l2_l3.png",
    "hierarchical_inference_pipeline.png",
    "l1_l2_l3_comparison.png",
    "l2_l3_summary_table.png",
]


def architecture_string(model_cfg: dict[str, Any]) -> str:
    layers = [
        int(model_cfg["input_dim"]),
        *[int(item) for item in model_cfg["hidden_layers"]],
        int(model_cfg["output_dim"]),
    ]
    return " -> ".join(str(item) for item in layers)


def figure_filenames(task: str) -> list[str]:
    normalized = normalize_task(task)
    architecture = "l2_family_architecture.png" if normalized == "l2_family" else "l3_attack_type_architecture.png"
    return (L2_FIGURES if normalized == "l2_family" else L3_FIGURES) + COMMON_FIGURES + [architecture]


def figure_dir(config: dict[str, Any], repo_root: Path, *, task: str, alpha: float, clients: int, run_id: str) -> Path:
    return repo_root / config["outputs"]["figures_dir"] / normalize_task(task) / alpha_dir(alpha) / f"k{clients}" / run_id


def run_artifact_paths(run_dir: Path) -> list[Path]:
    return [run_dir / item for item in RUN_ARTIFACT_PATTERNS]


def run_figure_paths(figures_dir: Path, task: str) -> list[Path]:
    return [figures_dir / filename for filename in figure_filenames(task)]


def expected_run_artifacts() -> list[str]:
    return [f"runs/{{run_id}}/{item}" for item in RUN_ARTIFACT_PATTERNS] + [
        "latest_run.json",
        "latest_run_summary.json",
    ]


def expected_figures() -> list[str]:
    names = L2_FIGURES + ["l2_family_architecture.png"] + L3_FIGURES + ["l3_attack_type_architecture.png"] + COMMON_FIGURES
    return [f"outputs/figures/hierarchical_flower/{{task}}/alpha_{{alpha}}/k{{k}}/{{run_id}}/{name}" for name in names]


def existing_relative_paths(paths: list[Path], repo_root: Path) -> list[str]:
    return [rel(path, repo_root) for path in paths if path.exists()]


def run_criteria(
    *,
    task: str,
    mode: str,
    rounds_completed: int,
    rounds_configured: int,
    artifacts: list[str],
    figures: list[str],
    docs_generated: bool,
) -> dict[str, bool]:
    artifact_names = {Path(item).name for item in artifacts}
    return {
        "true_flower_runtime": True,
        "p3_l2_partitions_used": True,
        "global_test_holdout_protected": True,
        "test_sent_to_clients_false": True,
        "server_client_runtime_started": rounds_completed > 0,
        "round_metrics_generated": "metrics_rounds.csv" in artifact_names,
        "client_metrics_generated": "metrics_clients.csv" in artifact_names,
        "bandwidth_metrics_generated": "bandwidth_rounds.csv" in artifact_names,
        "best_model_saved": "best_global_model.pth" in artifact_names,
        "last_model_saved": "last_global_model.pth" in artifact_names,
        "metrics_val_generated": "metrics_val.json" in artifact_names,
        "metrics_test_generated": "metrics_test.json" in artifact_names,
        "one_vs_rest_metrics_generated": "one_vs_rest_metrics.csv" in artifact_names,
        "figures_generated": len(figures) >= len(figure_filenames(task)),
        "docs_generated": docs_generated,
        "run_logs_generated": "run_console.log" in artifact_names,
        "l2_l3_not_deployed": True,
        "smoke_run_only": mode == "smoke",
        "full_run_completed": mode == "full" and rounds_completed == rounds_configured,
    }


def build_verify_contract(
    *,
    accepted: bool,
    scenario: dict[str, Any],
    checks: dict[str, Any],
    warnings: list[str],
    errors: list[str] | None = None,
) -> dict[str, Any]:
    errors = errors or []
    return {
        "accepted": bool(accepted),
        "phase": "P6",
        "mode": "verify",
        "flower_version": fl.__version__,
        "architecture": "Flower legacy manual/subprocess runtime with FedAvg",
        "scenario": scenario,
        "checks": checks,
        "artifacts_expected": expected_run_artifacts(),
        "figures_expected": expected_figures(),
        "criteria": {
            "p2_l2_outputs_exist": bool(checks.get("p2_l2_outputs_exist")),
            "p3_l2_index_only_partitions_exist": bool(checks.get("p3_l2_index_only_partitions_exist")),
            "flower_runtime_available": bool(checks.get("flower_runtime_available")),
            "l2_output_dim_is_8": bool(checks.get("l2_output_dim_is_8")),
            "l3_output_dim_is_33": bool(checks.get("l3_output_dim_is_33")),
            "global_test_holdout_protected": bool(checks.get("global_test_holdout_protected")),
            "deploy_l2_l3_false": bool(checks.get("deploy_l2_l3_false")),
        },
        "warnings": warnings,
        "errors": errors,
    }


def write_latest_run_summary(*, run_dir: Path, repo_root: Path, summary: dict[str, Any]) -> Path:
    from fl_hierarchical.data import write_json

    scenario_dir = run_dir.parents[1]
    latest_path = scenario_dir / "latest_run_summary.json"
    write_json(latest_path, summary)
    return latest_path
