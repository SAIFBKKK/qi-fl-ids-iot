"""Integration tests for P10 evidence pack artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
REPORTS = REPO_ROOT / "experiments/qi-fl-ids-iot-final/outputs/reports"
FIGURES_P10 = REPO_ROOT / "experiments/qi-fl-ids-iot-final/outputs/figures/p10"

REQUIRED_METHODS = {
    "P8 FedAvg + QGA L1",
    "P9 QIFA L1",
    "P9 QIFA + QGA L1",
    "P8-b L2 FedAvg + QGA Flower",
}

REQUIRED_FIGURES = [
    "p10_l1_macro_f1_comparison.png",
    "p10_l1_attack_recall_fpr_tradeoff.png",
    "p10_l1_bandwidth_comparison.png",
    "p10_method_ranking_table.png",
    "p10_final_architecture_summary.png",
]


@pytest.fixture(scope="module")
def csv_rows():
    path = REPORTS / "p10_global_comparison.csv"
    assert path.exists(), f"Missing: {path}"
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def manifest():
    path = REPORTS / "p10_evidence_manifest.json"
    assert path.exists(), f"Missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_csv_exists():
    assert (REPORTS / "p10_global_comparison.csv").exists()


def test_json_exists():
    assert (REPORTS / "p10_global_comparison.json").exists()


def test_manifest_exists():
    assert (REPORTS / "p10_evidence_manifest.json").exists()


def test_markdown_exists():
    assert (REPORTS / "p10_global_comparison_table.md").exists()


def test_at_least_8_methods(csv_rows):
    assert len(csv_rows) >= 8, f"Expected >= 8 methods, got {len(csv_rows)}"


def test_required_methods_present(csv_rows):
    methods_in_csv = {r["method"] for r in csv_rows}
    for method in REQUIRED_METHODS:
        assert method in methods_in_csv, f"Missing method: {method}"


def test_p8_qga_macro_f1_not_empty(csv_rows):
    row = next((r for r in csv_rows if r["method"] == "P8 FedAvg + QGA L1"), None)
    assert row is not None
    assert row.get("macro_f1") not in (None, "", "None"), "P8 macro_f1 is empty"
    assert float(row["macro_f1"]) > 0.9


def test_p9_qifa_macro_f1_not_empty(csv_rows):
    row = next((r for r in csv_rows if r["method"] == "P9 QIFA L1"), None)
    assert row is not None
    assert row.get("macro_f1") not in (None, "", "None")
    assert float(row["macro_f1"]) > 0.9


def test_p9_qifa_qga_macro_f1_not_empty(csv_rows):
    row = next((r for r in csv_rows if r["method"] == "P9 QIFA + QGA L1"), None)
    assert row is not None
    assert row.get("macro_f1") not in (None, "", "None")
    assert float(row["macro_f1"]) > 0.9


def test_accuracy_not_empty_for_main_methods(csv_rows):
    main = {"P4 Centralized L1", "P5 FedAvg L1", "P8 FedAvg + QGA L1"}
    for method in main:
        row = next((r for r in csv_rows if r["method"] == method), None)
        if row is not None:
            assert row.get("accuracy") not in (None, "", "None"), f"accuracy empty for {method}"


def test_accepted_field_present(csv_rows):
    for row in csv_rows:
        assert "accepted" in row, f"accepted field missing for {row.get('method')}"


def test_manifest_accepted(manifest):
    assert manifest.get("accepted") is True


def test_manifest_method_count(manifest):
    assert manifest.get("method_count", 0) >= 8


def test_required_figures_exist():
    for fig in REQUIRED_FIGURES:
        assert (FIGURES_P10 / fig).exists(), f"Missing figure: {fig}"
