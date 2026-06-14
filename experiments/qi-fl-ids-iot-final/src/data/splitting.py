"""Deterministic split helpers for P2 preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SPLIT_CODES: dict[str, int] = {"train": 0, "val": 1, "test": 2}
SPLIT_NAMES: dict[int, str] = {value: key for key, value in SPLIT_CODES.items()}


@dataclass(frozen=True)
class SplitResult:
    """Split assignments aligned with the provided row ids."""

    row_ids: np.ndarray
    assignments: np.ndarray
    counts: dict[str, int]
    counts_by_stratum: dict[str, dict[str, int]]


def _split_sizes(
    n_rows: int, train_ratio: float, val_ratio: float
) -> tuple[int, int, int]:
    train_count = int(np.floor(n_rows * train_ratio))
    val_count = int(np.floor(n_rows * val_ratio))
    test_count = n_rows - train_count - val_count
    return train_count, val_count, test_count


def stratified_train_val_test_split(
    row_ids: np.ndarray,
    strata: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> SplitResult:
    """Create deterministic 70/15/15-style assignments per stratum.

    The returned assignment array uses 0=train, 1=val and 2=test, aligned with
    the input row_ids. Each row_id is assigned exactly once.
    """

    if row_ids.shape[0] != strata.shape[0]:
        raise ValueError("row_ids and strata must have the same length")
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("split ratios must sum to 1.0")
    if row_ids.size == 0:
        raise ValueError("cannot split an empty dataset")

    rng = np.random.default_rng(random_seed)
    assignments = np.full(row_ids.shape[0], -1, dtype=np.int8)
    counts_by_stratum: dict[str, dict[str, int]] = {}

    for stratum in sorted(np.unique(strata).tolist()):
        local_positions = np.flatnonzero(strata == stratum)
        shuffled_positions = np.array(local_positions, copy=True)
        rng.shuffle(shuffled_positions)

        train_count, val_count, test_count = _split_sizes(
            shuffled_positions.size, train_ratio, val_ratio
        )
        train_end = train_count
        val_end = train_count + val_count

        assignments[shuffled_positions[:train_end]] = SPLIT_CODES["train"]
        assignments[shuffled_positions[train_end:val_end]] = SPLIT_CODES["val"]
        assignments[shuffled_positions[val_end:]] = SPLIT_CODES["test"]

        counts_by_stratum[str(stratum)] = {
            "train": int(train_count),
            "val": int(val_count),
            "test": int(test_count),
            "total": int(shuffled_positions.size),
        }

    if np.any(assignments < 0):
        raise RuntimeError("some rows were not assigned to a split")

    counts = {
        split_name: int(np.sum(assignments == split_code))
        for split_name, split_code in SPLIT_CODES.items()
    }
    return SplitResult(
        row_ids=row_ids,
        assignments=assignments,
        counts=counts,
        counts_by_stratum=counts_by_stratum,
    )


def assign_to_global(
    total_rows: int, row_ids: np.ndarray, local_assignments: np.ndarray
) -> np.ndarray:
    """Build a global split vector indexed by stable row_id."""

    if row_ids.shape[0] != local_assignments.shape[0]:
        raise ValueError("row_ids and assignments must have the same length")
    global_assignments = np.full(total_rows, -1, dtype=np.int8)
    global_assignments[row_ids] = local_assignments
    return global_assignments


def anti_leakage_report(split_assignments: np.ndarray) -> dict[str, Any]:
    """Report split exclusivity for a global assignment vector."""

    assigned = split_assignments >= 0
    counts = {
        split_name: int(np.sum(split_assignments == split_code))
        for split_name, split_code in SPLIT_CODES.items()
    }
    total_assigned = int(np.sum(assigned))
    no_overlap = sum(counts.values()) == total_assigned
    return {
        "anti_leakage_valid": bool(no_overlap),
        "anti_leakage_id": "row_id",
        "overlap_counts": {
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
        },
        "split_counts": counts,
        "total_assigned": total_assigned,
    }


def split_counts_by_key(
    split_assignments: np.ndarray,
    keys: np.ndarray,
    *,
    labels: dict[int, str] | None = None,
) -> dict[str, dict[str, int]]:
    """Count split assignments for each key value."""

    if split_assignments.shape[0] != keys.shape[0]:
        raise ValueError("split_assignments and keys must have the same length")

    report: dict[str, dict[str, int]] = {}
    assigned_mask = split_assignments >= 0
    for key in sorted(np.unique(keys[assigned_mask]).tolist()):
        key_int = int(key)
        key_mask = assigned_mask & (keys == key_int)
        name = labels.get(key_int, str(key_int)) if labels else str(key_int)
        report[name] = {
            split_name: int(np.sum(key_mask & (split_assignments == split_code)))
            for split_name, split_code in SPLIT_CODES.items()
        }
        report[name]["total"] = int(np.sum(key_mask))
    return report
