"""P8.1.5 QGA calibration and mask-selection utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from qga.config import repo_path


def get_qga_profiles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = config.get("qga_profiles", {})
    if not profiles:
        raise ValueError("qga_profiles section is missing")
    return {str(name): dict(payload) for name, payload in profiles.items()}


def calibration_seeds(config: dict[str, Any]) -> list[int]:
    return [int(seed) for seed in config.get("qga_calibration", {}).get("seeds", [42, 123, 2026])]


def build_profile_params(
    config: dict[str, Any],
    *,
    profile_name: str,
    seed: int,
    population_size: int | None = None,
    generations: int | None = None,
    max_samples_for_fitness: int | None = None,
) -> dict[str, Any]:
    profiles = get_qga_profiles(config)
    if profile_name not in profiles:
        raise ValueError(f"unknown QGA profile: {profile_name}")
    profile = profiles[profile_name]
    calibration = config.get("qga_calibration", {})
    params = dict(config["qga"])
    params["seed"] = int(seed)
    params["population_size"] = int(population_size or calibration.get("population_size", params["population_size"]))
    params["generations"] = int(generations or calibration.get("generations", params["generations"]))
    params["mutation_rate"] = float(calibration.get("mutation_rate", params["mutation_rate"]))
    params["max_samples_for_fitness"] = int(
        max_samples_for_fitness or calibration.get("max_samples_for_fitness", params["max_samples_for_fitness"])
    )
    params["min_features"] = int(profile["min_features"])
    params["max_features"] = int(profile["max_features"])
    params["weights"] = {
        "alpha_macro_f1": float(profile["alpha_macro_f1"]),
        "beta_attack_recall": float(profile["beta_attack_recall"]),
        "lambda_feature_penalty": float(profile["lambda_feature_penalty"]),
        "fpr_penalty": float(profile.get("fpr_penalty", 0.0)),
    }
    params["profile"] = profile_name
    params["mode"] = "calibration"
    return params


def mask_id(profile: str, seed: int) -> str:
    return f"{profile}_seed_{int(seed)}"


def engineering_score(
    *,
    mean_macro_f1: float,
    mean_attack_recall: float,
    mean_fpr: float,
    std_macro_f1: float,
    feature_ratio: float,
    weights: dict[str, float] | None = None,
) -> float:
    weights = weights or {
        "mean_macro_f1": 0.40,
        "mean_attack_recall": 0.25,
        "mean_fpr": -0.20,
        "std_macro_f1": -0.10,
        "feature_ratio": -0.05,
    }
    return float(
        weights["mean_macro_f1"] * mean_macro_f1
        + weights["mean_attack_recall"] * mean_attack_recall
        + weights["mean_fpr"] * mean_fpr
        + weights["std_macro_f1"] * std_macro_f1
        + weights["feature_ratio"] * feature_ratio
    )


def summarize_mask_stability(rows: list[dict[str, Any]], *, feature_names: list[str]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    total = 0
    for row in rows:
        features = row.get("selected_features", [])
        if isinstance(features, str):
            features = [item.strip() for item in features.split(";") if item.strip()]
        for feature in features:
            counts[str(feature)] += 1
        total += 1
    return [
        {
            "feature": feature,
            "selected_count": int(counts[feature]),
            "selection_frequency": float(counts[feature] / max(total, 1)),
        }
        for feature in feature_names
    ]


def group_validation_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mask_id"])].append(row)
    return dict(grouped)


def rank_masks_from_short_validation(
    rows: list[dict[str, Any]],
    *,
    score_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for current_mask_id, mask_rows in group_validation_rows(rows).items():
        macro = np.asarray([float(row["val_macro_f1"]) for row in mask_rows], dtype=np.float64)
        recall = np.asarray([float(row["val_attack_recall"]) for row in mask_rows], dtype=np.float64)
        fpr = np.asarray([float(row["val_fpr"]) for row in mask_rows], dtype=np.float64)
        feature_count = int(mask_rows[0]["features_count"])
        score = engineering_score(
            mean_macro_f1=float(macro.mean()),
            mean_attack_recall=float(recall.mean()),
            mean_fpr=float(fpr.mean()),
            std_macro_f1=float(macro.std()),
            feature_ratio=float(feature_count / 28.0),
            weights=score_weights,
        )
        ranked.append(
            {
                "mask_id": current_mask_id,
                "profile": mask_rows[0]["profile"],
                "seed": int(mask_rows[0]["seed"]),
                "features_count": feature_count,
                "mean_macro_f1": float(macro.mean()),
                "mean_attack_recall": float(recall.mean()),
                "mean_fpr": float(fpr.mean()),
                "std_macro_f1": float(macro.std()),
                "feature_ratio": float(feature_count / 28.0),
                "engineering_score": float(score),
                "scenario_count": len(mask_rows),
            }
        )
    return sorted(ranked, key=lambda row: (row["engineering_score"], -row["mean_fpr"], -row["features_count"]), reverse=True)


def profile_sweep_summary_path(config: dict[str, Any]) -> Path:
    return repo_path(config, "outputs.reports_dir") / "p8_qga_profile_sweep_summary.csv"


def short_validation_summary_path(config: dict[str, Any]) -> Path:
    return repo_path(config, "outputs.reports_dir") / "p8_qga_flower_short_validation.csv"
