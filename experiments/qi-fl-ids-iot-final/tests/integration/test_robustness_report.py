from __future__ import annotations

import sys
from pathlib import Path
import csv

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robustness.config import load_config
from robustness.report_builder import build_reports


CONFIG = Path("experiments/qi-fl-ids-iot-final/configs/robustness_l1.yaml")


def test_report_builder_runs_with_or_without_runs() -> None:
    result = build_reports(load_config(CONFIG))
    assert Path(result["reports"][0]).exists()
    assert Path(result["reports"][1]).exists()


def test_report_outputs_full_scientific_contract() -> None:
    build_reports(load_config(CONFIG))
    reports_dir = Path("experiments/qi-fl-ids-iot-final/outputs/reports")
    summary = reports_dir / "p10_robustness_summary.csv"
    full_summary = reports_dir / "p10_robustness_full_summary.csv"
    clean_vs_poisoned = reports_dir / "p10_robustness_clean_vs_poisoned.csv"
    findings = reports_dir / "p10_robustness_findings.md"
    assert summary.exists()
    assert full_summary.exists()
    assert clean_vs_poisoned.exists()

    with summary.open("r", encoding="utf-8", newline="") as file:
        all_rows = list(csv.DictReader(file))
    assert "run_type" in all_rows[0]
    assert "fpr" in all_rows[0]

    with full_summary.open("r", encoding="utf-8", newline="") as file:
        full_rows = list(csv.DictReader(file))
    assert {row["method"] for row in full_rows} == {"fedavg", "fedavg_qga", "qifa", "qifa_qga"}
    assert all(row["run_type"] == "full" for row in full_rows)
    assert all(row["macro_f1"] != "0.4786634460547504" for row in full_rows)
    assert "QIFA+QGA is the best global result" in findings.read_text(encoding="utf-8")
