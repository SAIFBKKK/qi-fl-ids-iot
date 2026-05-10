"""Verify P7 HeteroFL setup without training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from multitier_heterofl.config import list_scenarios, load_config, repo_path, tier_mapping_for_k
from multitier_heterofl.data import load_scenario, task_spec
from multitier_heterofl.report_builder import write_verify_outputs
from multitier_heterofl.summary_schema import verify_contract
from multitier_heterofl.supernet import build_supernet, tier_parameter_summary


def verify_setup(config_path: Path, *, write_outputs: bool = True) -> dict[str, Any]:
    repo_root = Path.cwd().resolve()
    config = load_config(config_path)
    warnings: list[str] = []
    errors: list[str] = []
    try:
        l1 = task_spec(config, repo_root, "l1")
        l2 = task_spec(config, repo_root, "l2")
        scenario_l1 = load_scenario(config, repo_root, task="l1", alpha=0.5, clients=3)
        scenario_l2 = load_scenario(config, repo_root, task="l2", alpha=0.5, clients=3)
        supernet_ok = build_supernet(output_dim=2).count_parameters() > 0 and build_supernet(output_dim=l2.output_dim).count_parameters() > 0
    except Exception as exc:
        l1 = None
        l2 = None
        scenario_l1 = None
        scenario_l2 = None
        supernet_ok = False
        errors.append(str(exc))
    checks = {
        "config_loads": True,
        "l1_partitions_exist": scenario_l1 is not None,
        "l2_partitions_exist": scenario_l2 is not None,
        "l1_test_holdout_exists": repo_path(repo_root, config["inputs"]["l1_test_npz"]).exists(),
        "l2_test_holdout_exists": repo_path(repo_root, config["inputs"]["l2_test_npz"]).exists(),
        "p4_metrics_exist": repo_path(repo_root, config["inputs"]["p4_l1_metrics"]).exists(),
        "p5_grid_summary_exists": repo_path(repo_root, config["inputs"]["p5_grid_summary"]).exists(),
        "p6_summary_exists": repo_path(repo_root, config["inputs"]["p6_hierarchical_summary"]).exists(),
        "run_l3_false": bool(config["tasks"]["run_l3"]) is False,
        "tier_mapping_k3_k4_k5": all(tier_mapping_for_k(config, k) for k in [3, 4, 5]),
        "supernet_builds": supernet_ok,
        "l1_output_dim_is_2": bool(l1 and l1.output_dim == 2),
        "l2_output_dim_is_8": bool(l2 and l2.output_dim == 8),
    }
    accepted = all(checks.values()) and not errors
    summary = verify_contract(accepted=accepted, checks=checks, warnings=warnings, errors=errors)
    summary["scenarios"] = [{"alpha": alpha, "clients": k} for alpha, k in list_scenarios(config)]
    summary["tier_mapping"] = {f"k{k}": tier_mapping_for_k(config, k) for k in [3, 4, 5]}
    summary["models"] = {
        "l1": tier_parameter_summary(2),
        "l2": tier_parameter_summary(8),
    }
    if write_outputs:
        summary["generated_files"] = write_verify_outputs(repo_root=repo_root, config=config, summary=summary)
    return summary
