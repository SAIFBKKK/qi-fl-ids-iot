"""Calibration helpers for P8-b QGA L2."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def get_profiles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(name): dict(payload) for name, payload in config.get("qga_l2_profiles", {}).items()}


def get_seeds(config: dict[str, Any]) -> list[int]:
    return [int(seed) for seed in config.get("calibration", {}).get("seeds", [42])]


def mask_id(profile: str, seed: int) -> str:
    return f"{profile}_seed_{int(seed)}"


def random_mask(rng: np.random.Generator, *, min_features: int, max_features: int, total_features: int = 28) -> np.ndarray:
    count = int(rng.integers(int(min_features), int(max_features) + 1))
    indices = rng.choice(total_features, size=count, replace=False)
    mask = np.zeros(total_features, dtype=np.int8)
    mask[indices] = 1
    return mask


def engineering_score(
    *,
    mean_macro_f1: float,
    mean_macro_recall: float,
    mean_macro_fpr: float,
    std_macro_f1: float,
    feature_ratio: float,
    weights: dict[str, float] | None = None,
) -> float:
    resolved = weights or {
        "mean_macro_f1": 0.45,
        "mean_macro_recall": 0.25,
        "mean_macro_fpr": -0.15,
        "std_macro_f1": -0.10,
        "feature_ratio": -0.05,
    }
    return float(
        resolved["mean_macro_f1"] * mean_macro_f1
        + resolved["mean_macro_recall"] * mean_macro_recall
        + resolved["mean_macro_fpr"] * mean_macro_fpr
        + resolved["std_macro_f1"] * std_macro_f1
        + resolved["feature_ratio"] * feature_ratio
    )


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _valid_short_flower_row(row: dict[str, Any]) -> bool:
    rounds = _as_float(row.get("rounds"))
    return (
        rounds is not None
        and int(round(rounds)) == 5
        and _as_bool(row.get("true_flower_runtime"))
        and not _as_bool(row.get("test_sent_to_clients"))
    )


def rank_masks(
    rows: list[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
    require_short_flower: bool = False,
    min_scenarios: int = 1,
    return_warnings: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    warnings: dict[str, Any] = {
        "ignored_invalid_metric_rows": {"count": 0, "rows": []},
        "ignored_incomplete_masks": {"count": 0, "masks": []},
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if require_short_flower and not _valid_short_flower_row(row):
            warnings["ignored_invalid_metric_rows"]["count"] += 1
            warnings["ignored_invalid_metric_rows"]["rows"].append(
                {"mask_id": row.get("mask_id"), "run_id": row.get("run_id"), "reason": "not_true_flower_5_round_validation"}
            )
            continue
        parsed = {
            "val_macro_f1": _as_float(row.get("val_macro_f1")),
            "val_macro_recall": _as_float(row.get("val_macro_recall")),
            "val_macro_fpr": _as_float(row.get("val_macro_fpr")),
            "features_count": _as_float(row.get("features_count")),
        }
        if any(value is None for value in parsed.values()):
            warnings["ignored_invalid_metric_rows"]["count"] += 1
            warnings["ignored_invalid_metric_rows"]["rows"].append(
                {"mask_id": row.get("mask_id"), "run_id": row.get("run_id"), "reason": "missing_or_invalid_required_metric"}
            )
            continue
        normalized = dict(row)
        normalized.update(parsed)
        grouped[str(row["mask_id"])].append(normalized)
    ranking: list[dict[str, Any]] = []
    for current_mask_id, items in grouped.items():
        macro_f1 = np.asarray([float(row["val_macro_f1"]) for row in items], dtype=float)
        macro_recall = np.asarray([float(row["val_macro_recall"]) for row in items], dtype=float)
        macro_fpr = np.asarray([float(row["val_macro_fpr"]) for row in items], dtype=float)
        features = int(float(items[0]["features_count"]))
        score = engineering_score(
            mean_macro_f1=float(macro_f1.mean()),
            mean_macro_recall=float(macro_recall.mean()),
            mean_macro_fpr=float(macro_fpr.mean()),
            std_macro_f1=float(macro_f1.std()),
            feature_ratio=features / 28.0,
            weights=weights,
        )
        ranking.append(
            {
                "mask_id": current_mask_id,
                "profile": items[0]["profile"],
                "seed": int(float(items[0]["seed"])),
                "features_count": features,
                "scenario_count": len(items),
                "mean_macro_f1": float(macro_f1.mean()),
                "mean_macro_recall": float(macro_recall.mean()),
                "mean_macro_fpr": float(macro_fpr.mean()),
                "std_macro_f1": float(macro_f1.std()),
                "feature_ratio": features / 28.0,
                "engineering_score": score,
            }
        )
    ranking = sorted(ranking, key=lambda row: float(row["engineering_score"]), reverse=True)
    if min_scenarios > 1:
        complete = []
        for row in ranking:
            if int(row["scenario_count"]) >= int(min_scenarios):
                complete.append(row)
            else:
                warnings["ignored_incomplete_masks"]["count"] += 1
                warnings["ignored_incomplete_masks"]["masks"].append(row)
        ranking = complete
    if return_warnings:
        return ranking, warnings
    return ranking
