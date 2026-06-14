"""Build P11 FedTN/MPS reports from dry-run and evaluation summaries."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .config import read_json, resolve, write_json
from .plotting import plot_summary


FIELDS = [
    "base_model",
    "rank",
    "run_id",
    "dry_run",
    "dense_num_parameters",
    "compressed_num_parameters",
    "parameter_reduction_ratio",
    "dense_model_size_bytes",
    "compressed_model_size_bytes",
    "compression_ratio",
    "dense_bandwidth_total_bytes",
    "compressed_bandwidth_total_bytes",
    "bandwidth_reduction_ratio",
    "macro_f1",
    "attack_recall",
    "fpr",
    "accepted",
]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def collect_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = resolve(config["outputs"]["run_dir"])
    rows: list[dict[str, Any]] = []
    for summary_path in root.glob("**/artifacts/run_summary.json"):
        summary = read_json(summary_path)
        comp = summary.get("compression", {})
        metrics = summary.get("test", {})
        rows.append(
            {
                "base_model": summary.get("base_model", ""),
                "rank": comp.get("rank", ""),
                "run_id": summary.get("run_id", ""),
                "dry_run": summary.get("dry_run", True),
                "dense_num_parameters": comp.get("dense_num_parameters", ""),
                "compressed_num_parameters": comp.get("compressed_num_parameters", ""),
                "parameter_reduction_ratio": comp.get("parameter_reduction_ratio", ""),
                "dense_model_size_bytes": comp.get("dense_model_size_bytes", ""),
                "compressed_model_size_bytes": comp.get("compressed_model_size_bytes", ""),
                "compression_ratio": comp.get("compression_ratio", ""),
                "dense_bandwidth_total_bytes": comp.get("dense_bandwidth_total_bytes", ""),
                "compressed_bandwidth_total_bytes": comp.get("compressed_bandwidth_total_bytes", ""),
                "bandwidth_reduction_ratio": comp.get("bandwidth_reduction_ratio", ""),
                "macro_f1": metrics.get("macro_f1", ""),
                "attack_recall": metrics.get("attack_recall", ""),
                "fpr": metrics.get("fpr", ""),
                "accepted": summary.get("accepted", False),
            }
        )
    rows.sort(key=lambda row: (str(row["base_model"]), int(row["rank"] or 0), str(row["run_id"])))
    return rows


def build_reports(config: dict[str, Any]) -> dict[str, Any]:
    reports_dir = resolve(config["outputs"]["reports_dir"])
    figures_dir = resolve(config["outputs"]["figures_dir"])
    rows = collect_rows(config)
    csv_path = reports_dir / "p11_fedtn_mps_summary.csv"
    json_path = reports_dir / "p11_fedtn_mps_summary.json"
    table_path = reports_dir / "p11_fedtn_mps_table.md"
    findings_path = reports_dir / "p11_fedtn_mps_findings.md"
    _write_csv(csv_path, rows)
    warnings = [] if rows else ["No FedTN/MPS runs found yet."]
    if any(str(row.get("dry_run")) == "True" or row.get("dry_run") is True for row in rows):
        warnings.append("Dry-run rows contain structural compression estimates only; checkpoint evaluation is pending.")
    write_json(json_path, {"rows": rows, "warnings": warnings})
    header = "| base_model | rank | compressed_params | model_size_bytes | compression_ratio | bandwidth_reduction |\n|---|---:|---:|---:|---:|---:|\n"
    lines = [
        f"| {row['base_model']} | {row['rank']} | {row['compressed_num_parameters']} | {row['compressed_model_size_bytes']} | {row['compression_ratio']} | {row['bandwidth_reduction_ratio']} |"
        for row in rows
    ]
    table_path.write_text(header + "\n".join(lines) + ("\n" if lines else "| pending |  |  |  |  |  |\n"), encoding="utf-8")
    best = min(rows, key=lambda row: float(row.get("compression_ratio") or 999.0)) if rows else None
    best_text = (
        f"{best['base_model']} rank={best['rank']} compression_ratio={best['compression_ratio']}"
        if best
        else "pending until at least one dry-run or evaluation is available"
    )
    findings_path.write_text(
        "# P11 FedTN/MPS Findings\n\n"
        "P11 evaluates tensor-network-inspired low-rank compression for L1 QGA models.\n\n"
        f"Available runs: {len(rows)}.\n\n"
        f"Best structural compression row: {best_text}.\n\n"
        "The L1 QGA model is already small, so gains are expected to be moderate and rank-sensitive. "
        "Checkpoint-based accuracy evaluation remains required before using compressed models as final evidence.\n",
        encoding="utf-8",
    )
    figures = plot_summary(rows, figures_dir)
    return {"rows": len(rows), "reports": [str(csv_path), str(json_path), str(table_path), str(findings_path)], "figures": [str(path) for path in figures]}
