"""Ablation report generation for QIFA."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from qifa.config import load_json, repo_path, write_json


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_p5_baseline(config: dict[str, Any]) -> dict[str, Any]:
    for row in _read_csv_rows(repo_path(config, "inputs.p5_grid_summary")):
        if _to_float(row.get("alpha")) == 0.5 and int(float(row.get("clients", 0))) == 3:
            return {
                "method": "P5 FedAvg baseline",
                "features_count": 28,
                "macro_f1": _to_float(row.get("macro_f1")),
                "weighted_f1": _to_float(row.get("weighted_f1")),
                "attack_recall": _to_float(row.get("attack_recall")),
                "fpr": _to_float(row.get("fpr")),
                "accuracy": _to_float(row.get("accuracy")),
                "model_size_bytes": int(float(row.get("model_size_bytes", 0))),
                "bandwidth_total_bytes": int(float(row.get("bandwidth_total_bytes", 0))),
                "aggregation_type": "FedAvg",
                "variant": "",
                "gamma": "",
                "alpha": 0.5,
                "clients": 3,
                "rounds": int(float(row.get("rounds", 30))),
                "true_flower_runtime": False,
                "accepted": True,
            }
    return {"method": "P5 FedAvg baseline", "accepted": False}


def _find_p8_baseline(config: dict[str, Any]) -> dict[str, Any]:
    for row in _read_csv_rows(repo_path(config, "inputs.p8_ablation_summary")):
        if row.get("method") == "P8 FedAvg + QGA Flower":
            return {
                "method": row["method"],
                "features_count": int(float(row.get("features_count", 0))),
                "macro_f1": _to_float(row.get("macro_f1")),
                "weighted_f1": _to_float(row.get("weighted_f1")),
                "attack_recall": _to_float(row.get("attack_recall")),
                "fpr": _to_float(row.get("fpr")),
                "accuracy": _to_float(row.get("accuracy")),
                "model_size_bytes": int(float(row.get("model_size_bytes", 0))),
                "bandwidth_total_bytes": int(float(row.get("bandwidth_total_bytes", 0))),
                "aggregation_type": "FedAvg+QGA",
                "variant": "",
                "gamma": "",
                "alpha": 0.5,
                "clients": 3,
                "rounds": int(float(row.get("rounds", 30) or 30)),
                "true_flower_runtime": str(row.get("true_flower_runtime")).lower() == "true",
                "accepted": str(row.get("accepted")).lower() == "true",
            }
    return {"method": "P8 FedAvg + QGA Flower", "accepted": False}


def _summary_to_row(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("test", {}).get("metrics", {})
    model = summary.get("model", {})
    communication = summary.get("communication", {})
    return {
        "method": "P9 QIFA + QGA Flower" if summary.get("feature_mode") == "qga_mask" or summary.get("use_qga_mask") else "P9 QIFA Flower",
        "features_count": int(summary.get("selected_features_count") or summary.get("dataset", {}).get("input_dim_selected", model.get("config", {}).get("input_dim", 28))),
        "macro_f1": _to_float(metrics.get("macro_f1")),
        "weighted_f1": _to_float(metrics.get("weighted_f1")),
        "attack_recall": _to_float(metrics.get("recall_attack")),
        "fpr": _to_float(metrics.get("FPR")),
        "accuracy": _to_float(metrics.get("accuracy")),
        "model_size_bytes": int(float(model.get("model_size_bytes", communication.get("model_size_bytes", 0)))),
        "bandwidth_total_bytes": int(float(communication.get("communication_cumulative_bytes", 0))),
        "aggregation_type": "QIFA-Hybrid",
        "variant": summary.get("variant"),
        "gamma": summary.get("gamma"),
        "alpha": summary.get("scenario", {}).get("alpha"),
        "clients": summary.get("scenario", {}).get("clients"),
        "rounds": summary.get("training", {}).get("rounds_completed", summary.get("scenario", {}).get("rounds")),
        "true_flower_runtime": bool(summary.get("true_flower_runtime")),
        "accepted": bool(summary.get("accepted")),
    }


def build_qifa_ablation_report(config: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p5 = _find_p5_baseline(config)
    p8 = _find_p8_baseline(config)
    rows.extend([p5, p8])
    summaries_by_mode: dict[bool, tuple[str, dict[str, Any]]] = {}
    for path in sorted(repo_path(config, "outputs.run_dir").glob("*/alpha_0.5/k3/variant_*/gamma_*/runs/*/artifacts/run_summary.json")):
        summary = load_json(path)
        key = bool(summary.get("use_qga_mask"))
        current = summaries_by_mode.get(key)
        if current is None or str(summary.get("run_id", "")) > current[0]:
            summaries_by_mode[key] = (str(summary.get("run_id", "")), summary)
    for _, summary in summaries_by_mode.values():
        rows.append(_summary_to_row(summary))
    p5_macro = _to_float(p5.get("macro_f1"))
    p8_macro = _to_float(p8.get("macro_f1"))
    for row in rows:
        macro = _to_float(row.get("macro_f1"))
        row["gap_macro_f1_vs_p5"] = "" if p5_macro is None or macro is None else macro - p5_macro
        row["gap_macro_f1_vs_p8"] = "" if p8_macro is None or macro is None else macro - p8_macro
    reports_dir = repo_path(config, "outputs.reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "p9_qifa_ablation_summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(reports_dir / "p9_qifa_ablation_summary.json", rows)
    md_lines = [
        "# P9 QIFA Ablation",
        "",
        "| method | features_count | macro_f1 | attack_recall | fpr | accuracy | aggregation_type | variant | gamma |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row.get('method','')} | {row.get('features_count','')} | {row.get('macro_f1','')} | {row.get('attack_recall','')} | {row.get('fpr','')} | {row.get('accuracy','')} | {row.get('aggregation_type','')} | {row.get('variant','')} | {row.get('gamma','')} |"
        )
    (reports_dir / "p9_qifa_ablation_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return rows
