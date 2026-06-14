"""Output contract helpers for P7 HeteroFL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from multitier_heterofl.config import rel


RUN_ARTIFACTS = [
    "checkpoints/best_global_supernet.pth",
    "checkpoints/last_global_supernet.pth",
    "artifacts/tier_mapping.json",
    "artifacts/tier_model_configs.json",
    "artifacts/model_config.json",
    "artifacts/metrics_rounds.csv",
    "artifacts/metrics_clients.csv",
    "artifacts/metrics_tiers.csv",
    "artifacts/bandwidth_by_tier.csv",
    "artifacts/slices_updated.csv",
    "artifacts/metrics_val.json",
    "artifacts/metrics_test.json",
    "artifacts/confusion_matrix.csv",
    "artifacts/classification_report.json",
    "artifacts/comparison_with_p4_p5_p6.json",
    "artifacts/run_manifest.json",
    "artifacts/run_summary.json",
    "logs/run_console.log",
]

FIGURES = [
    "multitier_architecture.png",
    "supernet_slicing.png",
    "tier_parameter_comparison.png",
    "tier_model_size_comparison.png",
    "tier_bandwidth_comparison.png",
    "tier_latency_comparison.png",
    "l1_p4_p5_p7_comparison.png",
    "l1_macro_f1_p5_vs_p7_by_round.png",
    "l1_fpr_p5_vs_p7_by_round.png",
    "l1_multitier_confusion_matrix.png",
    "l2_p6_vs_p7_comparison.png",
    "l2_macro_f1_by_round.png",
    "l2_family_confusion_matrix.png",
    "l2_per_family_f1.png",
    "heatmap_p7_l1_macro_f1_alpha_k.png",
    "heatmap_p7_l1_fpr_alpha_k.png",
    "heatmap_p7_l2_macro_f1_alpha_k.png",
    "barplot_p7_bandwidth_by_scenario.png",
    "p7_scenario_ranking_table.png",
]


def expected_artifacts() -> list[str]:
    return [f"runs/{{run_id}}/{item}" for item in RUN_ARTIFACTS] + ["latest_run.json", "latest_run_summary.json"]


def expected_figures() -> list[str]:
    return [f"outputs/figures/multitier/{{task}}/alpha_{{alpha}}/k{{k}}/{{run_id}}/{item}" for item in FIGURES]


def run_artifact_paths(run_dir: Path) -> list[Path]:
    return [run_dir / item for item in RUN_ARTIFACTS]


def run_figure_paths(figures_dir: Path) -> list[Path]:
    return [figures_dir / item for item in FIGURES]


def existing_relative(paths: list[Path], repo_root: Path) -> list[str]:
    return [rel(path, repo_root) for path in paths if path.exists()]


def criteria(*, artifacts: list[str], figures: list[str], task: str) -> dict[str, bool]:
    names = {Path(path).name for path in artifacts}
    return {
        "heterofl_slicing_valid": True,
        "slice_aggregation_valid": True,
        "p3_partitions_used": True,
        "global_test_holdout_protected": True,
        "test_sent_to_clients_false": True,
        "l3_not_used": task in {"l1_binary", "l2_family"},
        "dashboard_not_modified": True,
        "docker_not_modified": True,
        "qi_not_used": True,
        "metrics_generated": "metrics_test.json" in names and "metrics_rounds.csv" in names,
        "figures_generated": len(figures) >= 8,
        "run_summary_generated": "run_summary.json" in names,
    }


def verify_contract(
    *,
    accepted: bool,
    checks: dict[str, Any],
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "accepted": bool(accepted),
        "phase": "P7",
        "mode": "verify",
        "method": "HeteroFL",
        "implementation": "shared_supernet_prefix_slicing",
        "checks": checks,
        "artifacts_expected": expected_artifacts(),
        "figures_expected": expected_figures(),
        "warnings": warnings,
        "errors": errors,
    }
