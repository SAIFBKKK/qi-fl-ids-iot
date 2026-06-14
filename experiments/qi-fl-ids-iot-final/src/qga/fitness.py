"""Fitness function for P8 QGA."""

from __future__ import annotations

from typing import Any


def compute_qga_fitness(
    metrics: dict[str, Any],
    *,
    features_count: int,
    total_features: int,
    weights: dict[str, float],
) -> float:
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    attack_recall = float(
        metrics.get("recall_attack", metrics.get("attack_recall", metrics.get("recall", 0.0)))
    )
    fpr = float(metrics.get("FPR", metrics.get("fpr", 0.0)))
    feature_ratio = float(features_count) / float(total_features)
    return (
        float(weights["alpha_macro_f1"]) * macro_f1
        + float(weights["beta_attack_recall"]) * attack_recall
        - float(weights["lambda_feature_penalty"]) * feature_ratio
        - float(weights.get("fpr_penalty", 0.0)) * fpr
    )


def mask_constraint_status(
    *,
    features_count: int,
    min_features: int,
    max_features: int,
) -> dict[str, Any]:
    valid = int(min_features) <= int(features_count) <= int(max_features)
    return {
        "valid": valid,
        "features_count": int(features_count),
        "min_features": int(min_features),
        "max_features": int(max_features),
    }
