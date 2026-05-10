"""Tests for P8-b QGA L2 mask selection helpers."""

from __future__ import annotations

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga_l2.calibration import engineering_score, rank_masks


def test_l2_engineering_score_formula() -> None:
    score = engineering_score(mean_macro_f1=0.8, mean_macro_recall=0.7, mean_macro_fpr=0.2, std_macro_f1=0.1, feature_ratio=0.5)
    expected = 0.45 * 0.8 + 0.25 * 0.7 - 0.15 * 0.2 - 0.10 * 0.1 - 0.05 * 0.5
    assert abs(score - expected) < 1e-12


def test_rank_masks_requires_grouping_by_mask_id() -> None:
    rows = [
        {"mask_id": "a", "profile": "p", "seed": 42, "features_count": 12, "val_macro_f1": 0.8, "val_macro_recall": 0.7, "val_macro_fpr": 0.1},
        {"mask_id": "a", "profile": "p", "seed": 42, "features_count": 12, "val_macro_f1": 0.7, "val_macro_recall": 0.7, "val_macro_fpr": 0.1},
        {"mask_id": "b", "profile": "p", "seed": 123, "features_count": 10, "val_macro_f1": 0.1, "val_macro_recall": 0.1, "val_macro_fpr": 0.9},
    ]
    ranked = rank_masks(rows)
    assert ranked[0]["mask_id"] == "a"
    assert ranked[0]["scenario_count"] == 2


def test_rank_masks_ignores_empty_metric_rows() -> None:
    rows = [
        {"mask_id": "a", "profile": "p", "seed": 42, "features_count": 12, "val_macro_f1": 0.8, "val_macro_recall": "", "val_macro_fpr": 0.1},
        {"mask_id": "b", "profile": "p", "seed": 123, "features_count": 10, "val_macro_f1": 0.7, "val_macro_recall": 0.6, "val_macro_fpr": 0.2},
    ]
    ranked, warnings = rank_masks(rows, return_warnings=True)
    assert [row["mask_id"] for row in ranked] == ["b"]
    assert warnings["ignored_invalid_metric_rows"]["count"] == 1


def test_rank_masks_requires_true_flower_five_round_rows() -> None:
    valid_base = {
        "mask_id": "a",
        "profile": "p",
        "seed": 42,
        "features_count": 12,
        "val_macro_f1": 0.8,
        "val_macro_recall": 0.7,
        "val_macro_fpr": 0.1,
        "true_flower_runtime": True,
        "test_sent_to_clients": False,
    }
    rows = [
        {**valid_base, "alpha": 0.1, "rounds": 5},
        {**valid_base, "alpha": 0.5, "rounds": 5},
        {**valid_base, "alpha": 5.0, "rounds": 5},
        {**valid_base, "mask_id": "b", "rounds": 1},
        {**valid_base, "mask_id": "c", "rounds": 5, "true_flower_runtime": False},
        {**valid_base, "mask_id": "d", "rounds": 5, "test_sent_to_clients": True},
    ]
    ranked, warnings = rank_masks(rows, require_short_flower=True, min_scenarios=3, return_warnings=True)
    assert [row["mask_id"] for row in ranked] == ["a"]
    assert warnings["ignored_invalid_metric_rows"]["count"] == 3


def test_rank_masks_requires_minimum_scenario_count() -> None:
    rows = [
        {
            "mask_id": "a",
            "profile": "p",
            "seed": 42,
            "features_count": 12,
            "val_macro_f1": 0.8,
            "val_macro_recall": 0.7,
            "val_macro_fpr": 0.1,
        },
        {
            "mask_id": "a",
            "profile": "p",
            "seed": 42,
            "features_count": 12,
            "val_macro_f1": 0.7,
            "val_macro_recall": 0.7,
            "val_macro_fpr": 0.1,
        },
    ]
    ranked, warnings = rank_masks(rows, min_scenarios=3, return_warnings=True)
    assert ranked == []
    assert warnings["ignored_incomplete_masks"]["count"] == 1
