"""Build P8-b QGA L2 ablation report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga_l2.config import load_config, load_json, repo_path, write_json


def _latest_summary(config: dict) -> dict | None:
    path = repo_path(config, "outputs.qga_l2_flower_dir") / "alpha_0.5" / "k3" / "latest_run_summary.json"
    return load_json(path) if path.exists() else None


def _latest_p6_summary(config: dict) -> tuple[dict | None, str | None]:
    repo = repo_path(config)
    candidates = [
        repo / "experiments/qi-fl-ids-iot-final/outputs/hierarchical_flower/l2_family/alpha_0.5/k3/latest_run_summary.json",
        repo / "experiments/qi-fl-ids-iot-final/outputs/reports/hierarchical_flower_summary.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = load_json(path)
        if "summaries" in payload and payload["summaries"]:
            for item in payload["summaries"]:
                if item.get("task") == "l2_family" or item.get("phase") == "P6":
                    return item, str(path)
            return payload["summaries"][0], str(path)
        return payload, str(path)
    return None, None


def _metric(metrics: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = metrics.get(name)
        if value not in (None, ""):
            return value
    return ""


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_l2_model_size_bytes(input_dim: int, output_dim: int = 8) -> int:
    num_parameters = (input_dim * 128 + 128) + (128 * 64 + 64) + (64 * output_dim + output_dim)
    return int(num_parameters * 4)


def bandwidth_total_bytes(*, model_size_bytes: int, clients: int, rounds: int) -> int:
    return int(2 * int(clients) * int(model_size_bytes) * int(rounds))


def _rounds(summary: dict[str, Any], default: int = 30) -> int:
    return int(
        _to_float(
            summary.get("training", {}).get("rounds_completed")
            or summary.get("scenario", {}).get("rounds")
            or summary.get("training", {}).get("rounds_configured"),
            default,
        )
    )


def _clients(summary: dict[str, Any], default: int = 3) -> int:
    return int(_to_float(summary.get("scenario", {}).get("clients") or summary.get("scenario", {}).get("K"), default))


def _model_size(summary: dict[str, Any], *, input_dim: int) -> int:
    model_size = (
        summary.get("model", {}).get("model_size_bytes")
        or summary.get("communication", {}).get("model_size_bytes")
        or summary.get("best_validation_metrics", {}).get("model_size_bytes")
    )
    if model_size not in (None, ""):
        return int(_to_float(model_size))
    return infer_l2_model_size_bytes(input_dim)


def _summary_to_row(summary: dict[str, Any], *, method: str, features_count: int, baseline_macro_f1: float | None = None) -> dict[str, Any]:
    metrics = summary.get("test", {}).get("metrics") or summary.get("validation", {}).get("metrics", {})
    input_dim = int(features_count)
    size = _model_size(summary, input_dim=input_dim)
    clients = _clients(summary)
    rounds = _rounds(summary)
    macro_f1 = _metric(metrics, "macro_f1")
    row = {
        "method": method,
        "features_count": features_count,
        "feature_reduction_ratio": "" if features_count == 28 else 1 - float(features_count) / 28.0,
        "macro_f1": macro_f1,
        "weighted_f1": _metric(metrics, "weighted_f1"),
        "macro_recall": _metric(metrics, "macro_recall", "recall_macro"),
        "macro_precision": _metric(metrics, "macro_precision", "precision_macro"),
        "macro_fpr": _metric(metrics, "macro_fpr", "FPR_macro"),
        "accuracy": _metric(metrics, "accuracy"),
        "model_size_bytes": size,
        "bandwidth_total_bytes": bandwidth_total_bytes(model_size_bytes=size, clients=clients, rounds=rounds),
        "true_flower_runtime": summary.get("true_flower_runtime", summary.get("criteria", {}).get("true_flower_runtime", "")),
        "accepted": summary.get("accepted", ""),
    }
    if baseline_macro_f1 is not None and macro_f1 not in (None, ""):
        row["gap_macro_f1_vs_p6"] = float(macro_f1) - baseline_macro_f1
    else:
        row["gap_macro_f1_vs_p6"] = ""
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = _latest_summary(config)
    p6_summary, p6_source = _latest_p6_summary(config)
    rows = []
    warnings = []
    p6_macro_f1 = None
    p6_bandwidth = None
    if p6_summary:
        p6_row = _summary_to_row(p6_summary, method="P6 L2 Flower baseline", features_count=28)
        p6_macro_f1 = _to_float(p6_row.get("macro_f1"), 0.0)
        p6_bandwidth = _to_float(p6_row.get("bandwidth_total_bytes"), 0.0)
        p6_row["source"] = p6_source
        rows.append(p6_row)
    else:
        warnings.append("P6 L2 Flower baseline summary not found; baseline row is unavailable.")
    if summary:
        features = int(summary.get("qga", {}).get("selected_features_count") or summary.get("dataset", {}).get("input_dim_selected") or 28)
        p8_row = _summary_to_row(summary, method="P8-b L2 FedAvg + QGA Flower", features_count=features, baseline_macro_f1=p6_macro_f1)
        if p6_bandwidth:
            p8_row["bandwidth_reduction_ratio"] = 1 - float(p8_row["bandwidth_total_bytes"]) / float(p6_bandwidth)
        rows.append(p8_row)
    else:
        warnings.append("P8-b L2 FedAvg + QGA Flower latest_run_summary.json not found.")
    if warnings:
        rows.append(
            {
                "method": "WARNING",
                "features_count": "",
                "accepted": False,
                "warning": " | ".join(warnings),
            }
        )
    reports = repo_path(config, "outputs.reports_dir")
    csv_path = reports / "p8b_qga_l2_ablation_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in rows for key in row}), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(reports / "p8b_qga_l2_ablation_summary.json", rows)
    headers = ["method", "features_count", "macro_f1", "weighted_f1", "macro_recall", "macro_fpr", "accuracy", "model_size_bytes", "bandwidth_total_bytes", "gap_macro_f1_vs_p6"]
    lines = ["# P8-b QGA L2 Ablation", "", "Generated from latest available summaries.", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    if warnings:
        lines.extend(["", "## Warnings", "", *[f"- {warning}" for warning in warnings]])
    (reports / "p8b_qga_l2_ablation_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("P8-b QGA L2 ablation report built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
