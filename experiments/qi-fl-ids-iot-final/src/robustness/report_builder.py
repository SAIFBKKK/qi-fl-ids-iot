"""Build P10 robustness summary reports from available runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .config import read_json, resolve, write_json
from .plotting import plot_summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "alpha",
        "clients",
        "attack_type",
        "poison_rate",
        "poisoned_clients",
        "run_id",
        "macro_f1",
        "attack_recall",
        "FPR",
        "FNR",
        "accuracy",
        "robustness_score",
        "accepted",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def collect_run_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = resolve(config["outputs"]["run_dir"])
    rows: list[dict[str, Any]] = []
    for summary_path in root.glob("**/artifacts/run_summary.json"):
        summary = read_json(summary_path)
        scenario = summary.get("scenario", {})
        test = summary.get("test", {})
        rows.append(
            {
                "method": summary.get("method", ""),
                "alpha": scenario.get("alpha", ""),
                "clients": scenario.get("clients", ""),
                "attack_type": scenario.get("attack_type", ""),
                "poison_rate": scenario.get("poison_rate", ""),
                "poisoned_clients": scenario.get("poisoned_clients", ""),
                "run_id": summary.get("run_id", ""),
                "macro_f1": test.get("macro_f1", ""),
                "attack_recall": test.get("attack_recall", ""),
                "FPR": test.get("FPR", ""),
                "FNR": test.get("FNR", ""),
                "accuracy": test.get("accuracy", ""),
                "robustness_score": test.get("robustness_score", ""),
                "accepted": summary.get("accepted", False),
            }
        )
    rows.sort(key=lambda row: (str(row["method"]), str(row["attack_type"]), float(row["poison_rate"] or 0.0), str(row["run_id"])))
    return rows


def build_reports(config: dict[str, Any]) -> dict[str, Any]:
    reports_dir = resolve(config["outputs"]["reports_dir"])
    figures_dir = resolve(config["outputs"]["figures_dir"])
    rows = collect_run_rows(config)
    csv_path = reports_dir / "p10_robustness_summary.csv"
    json_path = reports_dir / "p10_robustness_summary.json"
    table_path = reports_dir / "p10_robustness_table.md"
    findings_path = reports_dir / "p10_robustness_findings.md"
    _write_csv(csv_path, rows)
    write_json(json_path, {"rows": rows, "warnings": [] if rows else ["No robustness runs found yet."]})
    header = "| method | attack | rate | macro_f1 | attack_recall | FPR | robustness_score |\n|---|---|---:|---:|---:|---:|---:|\n"
    lines = [
        f"| {row['method']} | {row['attack_type']} | {row['poison_rate']} | {row['macro_f1']} | {row['attack_recall']} | {row['FPR']} | {row['robustness_score']} |"
        for row in rows
    ]
    table_path.write_text(header + "\n".join(lines) + ("\n" if lines else "| pending | pending |  |  |  |  |  |\n"), encoding="utf-8")
    findings_path.write_text(
        "# P10 Robustness Findings\n\n"
        "This report aggregates defensive poisoning experiments for L1 binary IDS. "
        "Full conclusions should be written after manual full runs for FedAvg, FedAvg+QGA, QIFA, and QIFA+QGA.\n\n"
        f"Available runs: {len(rows)}.\n",
        encoding="utf-8",
    )
    figures = plot_summary(rows, figures_dir)
    return {
        "rows": len(rows),
        "reports": [str(csv_path), str(json_path), str(table_path), str(findings_path)],
        "figures": [str(path) for path in figures],
    }
