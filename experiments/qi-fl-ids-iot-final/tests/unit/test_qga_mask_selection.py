"""Tests for P8.1.5 QGA mask engineering score."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga.calibration import engineering_score, rank_masks_from_short_validation


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "src" / "scripts" / "08_1_5_select_best_qga_mask.py"
SPEC = importlib.util.spec_from_file_location("select_best_qga_mask", SCRIPT_PATH)
select_best_qga_mask = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(select_best_qga_mask)


def test_engineering_score_formula() -> None:
    score = engineering_score(
        mean_macro_f1=0.94,
        mean_attack_recall=0.95,
        mean_fpr=0.06,
        std_macro_f1=0.01,
        feature_ratio=9 / 28,
    )
    expected = 0.40 * 0.94 + 0.25 * 0.95 - 0.20 * 0.06 - 0.10 * 0.01 - 0.05 * (9 / 28)
    assert abs(score - expected) < 1e-12


def test_rank_masks_prefers_higher_engineering_score() -> None:
    rows = [
        {
            "mask_id": "a",
            "profile": "balanced",
            "seed": 42,
            "features_count": 9,
            "val_macro_f1": 0.94,
            "val_attack_recall": 0.95,
            "val_fpr": 0.06,
        },
        {
            "mask_id": "b",
            "profile": "compression",
            "seed": 123,
            "features_count": 8,
            "val_macro_f1": 0.80,
            "val_attack_recall": 0.90,
            "val_fpr": 0.20,
        },
    ]
    ranked = rank_masks_from_short_validation(rows)
    assert ranked[0]["mask_id"] == "a"


def test_short_validation_rows_mark_test_unused() -> None:
    rows = [
        {
            "mask_id": "a",
            "profile": "balanced",
            "seed": 42,
            "features_count": 9,
            "val_macro_f1": 0.94,
            "val_attack_recall": 0.95,
            "val_fpr": 0.06,
            "true_flower_runtime": True,
            "test_sent_to_clients": False,
            "test_evaluated": False,
        }
    ]
    assert rows[0]["true_flower_runtime"] is True
    assert rows[0]["test_sent_to_clients"] is False
    assert rows[0]["test_evaluated"] is False


def test_mask_selection_filters_stale_non_flower_and_test_leak_rows() -> None:
    rows = [
        {"mask_id": "stale", "rounds": "1", "true_flower_runtime": "True", "test_sent_to_clients": "False"},
        {"mask_id": "not_flower", "rounds": "5", "true_flower_runtime": "False", "test_sent_to_clients": "False"},
        {"mask_id": "test_leak", "rounds": "5", "true_flower_runtime": "True", "test_sent_to_clients": "True"},
        {"mask_id": "valid", "rounds": "5", "true_flower_runtime": "True", "test_sent_to_clients": "False"},
    ]
    filtered, warnings = select_best_qga_mask._filter_validation_rows(rows)
    assert [row["mask_id"] for row in filtered] == ["valid"]
    assert warnings["ignored_stale_short_runs"]["count"] == 1
    assert warnings["ignored_invalid_runtime_rows"]["count"] == 2


def test_incomplete_masks_are_removed_from_final_ranking() -> None:
    rows = [
        {
            "mask_id": "complete",
            "profile": "balanced",
            "seed": 42,
            "features_count": 9,
            "val_macro_f1": 0.90,
            "val_attack_recall": 0.90,
            "val_fpr": 0.10,
        },
        {
            "mask_id": "complete",
            "profile": "balanced",
            "seed": 42,
            "features_count": 9,
            "val_macro_f1": 0.91,
            "val_attack_recall": 0.90,
            "val_fpr": 0.10,
        },
        {
            "mask_id": "complete",
            "profile": "balanced",
            "seed": 42,
            "features_count": 9,
            "val_macro_f1": 0.92,
            "val_attack_recall": 0.90,
            "val_fpr": 0.10,
        },
        {
            "mask_id": "incomplete",
            "profile": "fpr_aware",
            "seed": 42,
            "features_count": 13,
            "val_macro_f1": 0.99,
            "val_attack_recall": 1.0,
            "val_fpr": 0.01,
        },
    ]
    raw_ranking = rank_masks_from_short_validation(rows)
    final_ranking = [row for row in raw_ranking if row["scenario_count"] >= 3]
    assert [row["mask_id"] for row in final_ranking] == ["complete"]
