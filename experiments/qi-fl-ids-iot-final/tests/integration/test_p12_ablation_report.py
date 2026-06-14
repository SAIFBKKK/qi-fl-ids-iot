from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("experiments/qi-fl-ids-iot-final/src/scripts/12_build_final_ablation_report.py")
REPORTS = Path("experiments/qi-fl-ids-iot-final/outputs/reports")
FIGURES = Path("experiments/qi-fl-ids-iot-final/outputs/figures/p12_ablation")


def _run_builder() -> None:
    subprocess.run([sys.executable, str(SCRIPT)], check=True)


def _read_rows() -> list[dict[str, str]]:
    with (REPORTS / "p12_global_ablation_summary.csv").open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_p12_outputs_and_key_methods_exist() -> None:
    _run_builder()
    assert (REPORTS / "p12_global_ablation_summary.csv").exists()
    assert (REPORTS / "p12_global_ablation_summary.json").exists()
    assert (REPORTS / "p12_final_findings.md").exists()

    rows = _read_rows()
    assert len(rows) >= 10
    methods = {row["method"] for row in rows}
    assert "P8 FedAvg + QGA L1" in methods
    assert "P9 QIFA L1" in methods
    assert "P9 QIFA + QGA L1" in methods
    assert "P10 QIFA + QGA poisoned" in methods
    assert any(method.startswith("P11 FedTN/MPS") for method in methods)

    p11_rows = [row for row in rows if row["phase"] == "P11"]
    assert p11_rows
    assert all(row["result_type"] in {"dry_run", "structural"} for row in p11_rows)

    measured_main = [
        row
        for row in rows
        if row["result_type"] == "measured"
        and row["method"]
        in {"P8 FedAvg + QGA L1", "P9 QIFA L1", "P9 QIFA + QGA L1", "P10 QIFA + QGA poisoned"}
    ]
    assert measured_main
    assert all(row["macro_f1"] for row in measured_main)

    manifest = json.loads((REPORTS / "p12_evaluation_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["figures"]) >= 9
    for name in [
        "p12_l1_macro_f1_comparison.png",
        "p12_l1_attack_recall_fpr_tradeoff.png",
        "p12_robustness_under_poisoning.png",
        "p12_compression_size_reduction.png",
        "p12_final_method_ranking_table.png",
    ]:
        assert (FIGURES / name).exists()
