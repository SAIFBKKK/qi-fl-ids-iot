"""Metric extraction helpers for P11 reports."""

from __future__ import annotations

from typing import Any


def baseline_metric_row(summary_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_f1": summary_row.get("macro_f1", ""),
        "attack_recall": summary_row.get("attack_recall", ""),
        "fpr": summary_row.get("fpr", ""),
        "accuracy": summary_row.get("accuracy", ""),
    }
