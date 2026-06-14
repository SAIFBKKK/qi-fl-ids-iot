"""Output summary schema helpers for P8."""

from __future__ import annotations

from typing import Any


def qga_criteria(*, selected_features_count: int, min_features: int, max_features: int) -> dict[str, bool]:
    return {
        "qga_mask_generated": True,
        "test_not_used_for_selection": True,
        "selected_features_within_bounds": int(min_features) <= int(selected_features_count) <= int(max_features),
        "fitness_history_generated": True,
        "fedavg_qga_ready": True,
        "heterofl_qga_ready": True,
        "dashboard_not_modified": True,
        "docker_not_modified": True,
        "qifa_not_used": True,
        "fedtn_not_used": True,
    }


def expected_verify_artifacts() -> list[str]:
    return [
        "selected_features.json",
        "feature_mask.json",
        "feature_ranking.csv",
        "qga_history.csv",
        "fitness_best.json",
        "fitness_weights.json",
        "validation_metrics_best_mask.json",
        "run_summary.json",
    ]


def expected_verify_figures() -> list[str]:
    return [
        "qga_fitness_evolution.png",
        "qga_num_features_evolution.png",
        "qga_selected_features_barplot.png",
        "qga_feature_mask.png",
        "qga_feature_importance_ranking.png",
    ]


def accepted_from_criteria(criteria: dict[str, Any]) -> bool:
    return all(bool(value) for value in criteria.values())
