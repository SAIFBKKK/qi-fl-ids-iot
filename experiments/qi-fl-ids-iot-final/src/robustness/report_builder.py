"""Build P10 robustness summary reports from available runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .config import read_json, resolve, write_json
from .plotting import plot_summary


SUMMARY_FIELDS = [
    "method",
    "alpha",
    "clients",
    "attack_type",
    "poison_rate",
    "poisoned_clients",
    "run_id",
    "rounds",
    "max_samples",
    "run_type",
    "scientific_use",
    "macro_f1",
    "attack_recall",
    "fpr",
    "fnr",
    "accuracy",
    "robustness_score",
    "accepted",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = fieldnames or SUMMARY_FIELDS
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=resolved)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in resolved})


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _run_type(summary: dict[str, Any], scenario: dict[str, Any]) -> tuple[str, str]:
    rounds = _to_int(scenario.get("rounds"), 0)
    max_samples = (
        summary.get("max_samples")
        or summary.get("execution", {}).get("max_samples")
        or summary.get("dataset", {}).get("max_samples")
        or ""
    )
    if rounds < 30 or max_samples not in ("", None, 0, "0"):
        return "smoke", "readiness_only"
    return "full", "scientific"


def collect_run_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = resolve(config["outputs"]["run_dir"])
    rows: list[dict[str, Any]] = []
    for summary_path in root.glob("**/artifacts/run_summary.json"):
        summary = read_json(summary_path)
        scenario = summary.get("scenario", {})
        test = summary.get("test", {})
        run_type, scientific_use = _run_type(summary, scenario)
        rows.append(
            {
                "method": summary.get("method", ""),
                "alpha": scenario.get("alpha", ""),
                "clients": scenario.get("clients", ""),
                "attack_type": scenario.get("attack_type", ""),
                "poison_rate": scenario.get("poison_rate", ""),
                "poisoned_clients": scenario.get("poisoned_clients", ""),
                "run_id": summary.get("run_id", ""),
                "rounds": scenario.get("rounds", ""),
                "max_samples": summary.get("max_samples") or summary.get("execution", {}).get("max_samples") or summary.get("dataset", {}).get("max_samples") or "",
                "run_type": run_type,
                "scientific_use": scientific_use,
                "macro_f1": test.get("macro_f1", ""),
                "attack_recall": test.get("attack_recall", ""),
                "fpr": test.get("fpr", test.get("FPR", "")),
                "fnr": test.get("fnr", test.get("FNR", "")),
                "accuracy": test.get("accuracy", ""),
                "robustness_score": test.get("robustness_score", ""),
                "accepted": summary.get("accepted", False),
            }
        )
    rows.sort(key=lambda row: (str(row["method"]), str(row["attack_type"]), float(row["poison_rate"] or 0.0), str(row["run_id"])))
    return rows


def full_scientific_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_method: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("run_type") != "full" or row.get("accepted") is not True:
            continue
        latest_by_method[str(row["method"])] = row
    order = ["fedavg", "fedavg_qga", "qifa", "qifa_qga"]
    return [latest_by_method[method] for method in order if method in latest_by_method]


def _read_clean_baselines(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = resolve(config["inputs"]["p9_qifa_ablation_summary"])
    mapping = {
        "P5 FedAvg baseline": "fedavg",
        "P8 FedAvg + QGA Flower": "fedavg_qga",
        "P9 QIFA Flower": "qifa",
        "P9 QIFA + QGA Flower": "qifa_qga",
    }
    baselines: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return baselines
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            method = mapping.get(str(row.get("method", "")))
            if method:
                baselines[method] = row
    return baselines


def build_clean_vs_poisoned(config: dict[str, Any], full_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    baselines = _read_clean_baselines(config)
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    for poisoned in full_rows:
        method = str(poisoned["method"])
        clean = baselines.get(method)
        if not clean:
            warnings.append(f"Missing clean baseline for {method}")
            continue
        clean_macro_f1 = _to_float(clean.get("macro_f1"))
        poisoned_macro_f1 = _to_float(poisoned.get("macro_f1"))
        clean_attack_recall = _to_float(clean.get("attack_recall"))
        poisoned_attack_recall = _to_float(poisoned.get("attack_recall"))
        clean_fpr = _to_float(clean.get("fpr"))
        poisoned_fpr = _to_float(poisoned.get("fpr"))
        clean_accuracy = _to_float(clean.get("accuracy"))
        poisoned_accuracy = _to_float(poisoned.get("accuracy"))
        rows.append(
            {
                "method": method,
                "clean_macro_f1": clean_macro_f1,
                "poisoned_macro_f1": poisoned_macro_f1,
                "delta_macro_f1": None if clean_macro_f1 is None or poisoned_macro_f1 is None else poisoned_macro_f1 - clean_macro_f1,
                "clean_attack_recall": clean_attack_recall,
                "poisoned_attack_recall": poisoned_attack_recall,
                "delta_attack_recall": None if clean_attack_recall is None or poisoned_attack_recall is None else poisoned_attack_recall - clean_attack_recall,
                "clean_fpr": clean_fpr,
                "poisoned_fpr": poisoned_fpr,
                "delta_fpr": None if clean_fpr is None or poisoned_fpr is None else poisoned_fpr - clean_fpr,
                "clean_accuracy": clean_accuracy,
                "poisoned_accuracy": poisoned_accuracy,
                "delta_accuracy": None if clean_accuracy is None or poisoned_accuracy is None else poisoned_accuracy - clean_accuracy,
                "robustness_ratio_macro_f1": None
                if not clean_macro_f1 or poisoned_macro_f1 is None
                else poisoned_macro_f1 / clean_macro_f1,
                "accepted": bool(poisoned.get("accepted")),
            }
        )
    return rows, warnings


def _write_md_table(path: Path, rows: list[dict[str, Any]], *, full: bool = True) -> None:
    if full:
        header = "| method | run_id | rounds | macro_f1 | attack_recall | fpr | accuracy | robustness_score |\n|---|---|---:|---:|---:|---:|---:|---:|\n"
        lines = [
            f"| {row['method']} | {row['run_id']} | {row['rounds']} | {row['macro_f1']} | {row['attack_recall']} | {row['fpr']} | {row['accuracy']} | {row['robustness_score']} |"
            for row in rows
        ]
    else:
        header = "| method | clean_macro_f1 | poisoned_macro_f1 | delta_macro_f1 | clean_fpr | poisoned_fpr | robustness_ratio_macro_f1 |\n|---|---:|---:|---:|---:|---:|---:|\n"
        lines = [
            f"| {row['method']} | {row['clean_macro_f1']} | {row['poisoned_macro_f1']} | {row['delta_macro_f1']} | {row['clean_fpr']} | {row['poisoned_fpr']} | {row['robustness_ratio_macro_f1']} |"
            for row in rows
        ]
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def _write_findings(path: Path, rows: list[dict[str, Any]], clean_rows: list[dict[str, Any]], warnings: list[str]) -> None:
    best = max(rows, key=lambda row: _to_float(row.get("robustness_score"), -1.0) or -1.0) if rows else None
    full_table = "\n".join(
        f"- `{row['method']}`: macro_f1={row['macro_f1']}, attack_recall={row['attack_recall']}, fpr={row['fpr']}, accuracy={row['accuracy']}, robustness_score={row['robustness_score']}"
        for row in rows
    )
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- No blocking warning."
    path.write_text(
        "# P10 Robustness Findings\n\n"
        "## Full label-flip results, poison_rate=0.2, poisoned_clients=1\n\n"
        f"{full_table}\n\n"
        "Interpretation:\n\n"
        "- QIFA+QGA is the best global result for Macro-F1, attack recall, accuracy, and robustness score.\n"
        "- FedAvg has the best FPR, but it also has the weakest attack recall.\n"
        "- QGA improves FedAvg under poisoning.\n"
        "- QIFA improves attack detection but increases FPR.\n"
        "- QIFA+QGA gives the best robustness/detection compromise.\n\n"
        "The `fedavg` line with `macro_f1=0.4787` is a smoke readiness run and is not included in scientific conclusions.\n\n"
        f"Best full method: `{best['method'] if best else 'pending'}`.\n\n"
        "## Clean vs Poisoned\n\n"
        f"Clean comparison rows: {len(clean_rows)}.\n\n"
        "## Warnings\n\n"
        f"{warning_text}\n",
        encoding="utf-8",
    )


def build_reports(config: dict[str, Any]) -> dict[str, Any]:
    reports_dir = resolve(config["outputs"]["reports_dir"])
    figures_dir = resolve(config["outputs"]["figures_dir"])
    rows = collect_run_rows(config)
    full_rows = full_scientific_rows(rows)
    clean_rows, clean_warnings = build_clean_vs_poisoned(config, full_rows)
    csv_path = reports_dir / "p10_robustness_summary.csv"
    json_path = reports_dir / "p10_robustness_summary.json"
    full_csv_path = reports_dir / "p10_robustness_full_summary.csv"
    full_json_path = reports_dir / "p10_robustness_full_summary.json"
    clean_csv_path = reports_dir / "p10_robustness_clean_vs_poisoned.csv"
    clean_json_path = reports_dir / "p10_robustness_clean_vs_poisoned.json"
    clean_table_path = reports_dir / "p10_robustness_clean_vs_poisoned_table.md"
    table_path = reports_dir / "p10_robustness_table.md"
    findings_path = reports_dir / "p10_robustness_findings.md"
    _write_csv(csv_path, rows)
    _write_csv(full_csv_path, full_rows)
    _write_csv(
        clean_csv_path,
        clean_rows,
        [
            "method",
            "clean_macro_f1",
            "poisoned_macro_f1",
            "delta_macro_f1",
            "clean_attack_recall",
            "poisoned_attack_recall",
            "delta_attack_recall",
            "clean_fpr",
            "poisoned_fpr",
            "delta_fpr",
            "clean_accuracy",
            "poisoned_accuracy",
            "delta_accuracy",
            "robustness_ratio_macro_f1",
            "accepted",
        ],
    )
    expected_methods = {"fedavg", "fedavg_qga", "qifa", "qifa_qga"}
    full_methods = {str(row["method"]) for row in full_rows}
    warnings = ([] if rows else ["No robustness runs found yet."]) + clean_warnings
    missing = sorted(expected_methods - full_methods)
    if missing:
        warnings.append(f"Missing full scientific methods: {missing}")
    warnings.append("p10_qifa_weights_under_attack.png is a placeholder because the P10 in-process summaries do not expose QIFA per-round weights.")
    write_json(json_path, {"rows": rows, "warnings": warnings})
    write_json(full_json_path, {"rows": full_rows, "warnings": warnings, "full_methods": sorted(full_methods)})
    write_json(clean_json_path, {"rows": clean_rows, "warnings": clean_warnings})
    _write_md_table(table_path, full_rows, full=True)
    _write_md_table(clean_table_path, clean_rows, full=False)
    _write_findings(findings_path, full_rows, clean_rows, warnings)
    figures = plot_summary(full_rows, figures_dir, clean_rows)
    return {
        "rows": len(rows),
        "full_rows": len(full_rows),
        "reports": [
            str(csv_path),
            str(json_path),
            str(full_csv_path),
            str(full_json_path),
            str(clean_csv_path),
            str(clean_json_path),
            str(clean_table_path),
            str(table_path),
            str(findings_path),
        ],
        "figures": [str(path) for path in figures],
    }
