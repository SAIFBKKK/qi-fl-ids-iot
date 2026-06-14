from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


FINAL_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = FINAL_DIR.parents[1]
REPORTS_DIR = FINAL_DIR / "outputs" / "reports"
DASHBOARD_DIR = FINAL_DIR / "dashboard"
DASHBOARD_DATA_DIR = DASHBOARD_DIR / "data"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _collect_figures(directory: Path) -> list[dict[str, str]]:
    if not directory.exists():
        return []
    return [
        {"name": path.name, "path": _relative(path)}
        for path in sorted(directory.glob("*.png"))
    ]


def _method_row(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    return next((row for row in rows if row.get("method") == method), {})


def build_dashboard_summary() -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    registry = _read_json(DASHBOARD_DIR / "model_registry.json")
    p12_rows = _read_csv(REPORTS_DIR / "p12_global_ablation_summary.csv")
    p10_rows = _read_csv(REPORTS_DIR / "p10_robustness_full_summary.csv")
    p11_rows = _read_csv(REPORTS_DIR / "p11_fedtn_mps_summary.csv")
    evaluation_rows = _read_csv(REPORTS_DIR / "p13_dashboard_model_evaluation.csv")
    qga_mask = _read_json(FINAL_DIR / "outputs" / "qga_feature_selection" / "final_selected_mask" / "selected_features.json")

    if not p12_rows:
        warnings.append("p12_global_ablation_summary_missing")
    if not p10_rows:
        warnings.append("p10_robustness_full_summary_missing")
    if not p11_rows:
        warnings.append("p11_fedtn_mps_summary_missing")
    if not evaluation_rows:
        warnings.append("p13_model_evaluation_not_yet_generated")

    l1_methods = {
        "P5 FedAvg L1",
        "P8 FedAvg + QGA L1",
        "P9 QIFA L1",
        "P9 QIFA + QGA L1",
    }
    l1_rows = [row for row in p12_rows if row.get("method") in l1_methods]
    recommended = _method_row(p12_rows, "P8 FedAvg + QGA L1")
    qifa = _method_row(p12_rows, "P9 QIFA L1")
    qifa_qga = _method_row(p12_rows, "P9 QIFA + QGA L1")
    p10_qifa_qga = next((row for row in p10_rows if row.get("method") == "qifa_qga"), {})
    p11_rank8 = next((row for row in p11_rows if row.get("rank") == "8"), p11_rows[0] if p11_rows else {})

    if p11_rank8 and not p11_rank8.get("macro_f1"):
        warnings.append("p11_fedtn_mps_rank8_is_dry_run_no_measured_macro_f1")

    figure_groups = {
        "p12_ablation": _collect_figures(FINAL_DIR / "outputs" / "figures" / "p12_ablation"),
        "p10_robustness": _collect_figures(FINAL_DIR / "outputs" / "figures" / "robustness_l1"),
        "p11_compression": _collect_figures(FINAL_DIR / "outputs" / "figures" / "fedtn_mps_l1"),
    }

    summary = {
        "project": {
            "name": "Quantum-Inspired Federated Learning IDS for IoT",
            "phase": "P13",
            "task": "L1 binary IDS dashboard",
            "branch": "final/quantum-inspired-fl-iot-ids-final",
            "latest_input_tag": "final-v1.4-ablation-evaluation-reports",
            "deployment_model_recommendation": "P8 FedAvg + QGA L1",
        },
        "modes": {
            "evidence": "Uses validated P12/P10/P11 reports without retraining.",
            "evaluation": "Recomputes L1 test metrics only when compatible local checkpoints are available.",
            "demo_replay": "Optional future replay mode for P14; not required for P13.",
        },
        "recommended_model": {
            "model_id": "p8_fedavg_qga_l1",
            "method": "P8 FedAvg + QGA L1",
            "selected_mask_id": qga_mask.get("mask_id", "conservative_seed_42"),
            "features_count": qga_mask.get("selected_features_count", recommended.get("features_count", "12")),
            "selected_features": qga_mask.get("selected_features", []),
            "macro_f1": recommended.get("macro_f1", ""),
            "attack_recall": recommended.get("attack_recall", ""),
            "fpr": recommended.get("fpr", ""),
            "accuracy": recommended.get("accuracy", ""),
            "bandwidth_total_bytes": recommended.get("bandwidth_total_bytes", ""),
            "recommended_use_case": "Best production L1 tradeoff: high Macro-F1, QGA feature reduction, true Flower runtime evidence.",
        },
        "alternatives": [
            {
                "model_id": "p9_qifa_l1",
                "method": "P9 QIFA L1",
                "macro_f1": qifa.get("macro_f1", ""),
                "attack_recall": qifa.get("attack_recall", ""),
                "fpr": qifa.get("fpr", ""),
                "use_case": "Lowest FPR alternative.",
            },
            {
                "model_id": "p9_qifa_qga_l1",
                "method": "P9 QIFA + QGA L1",
                "macro_f1": qifa_qga.get("macro_f1", ""),
                "attack_recall": qifa_qga.get("attack_recall", ""),
                "fpr": qifa_qga.get("fpr", ""),
                "use_case": "Best attack recall / robustness alternative.",
            },
        ],
        "l1_comparison": l1_rows,
        "model_registry": registry.get("models", []),
        "model_evaluations": evaluation_rows,
        "robustness": {
            "scenario": "label_flip, poison_rate=0.2, poisoned_clients=1, alpha=0.5, K=3, rounds=30",
            "best_method": "qifa_qga",
            "best_row": p10_qifa_qga,
            "full_results": p10_rows,
        },
        "compression": {
            "method": "P11 FedTN/MPS rank 8 dry-run",
            "result_type": "dry_run",
            "row": p11_rank8,
            "warning": "Structural compression estimates only; no measured Macro-F1 because checkpoint evaluation was unavailable.",
        },
        "figures": figure_groups,
        "evidence_paths": {
            "p12_global": _relative(REPORTS_DIR / "p12_global_ablation_summary.csv"),
            "p10_robustness": _relative(REPORTS_DIR / "p10_robustness_full_summary.csv"),
            "p11_compression": _relative(REPORTS_DIR / "p11_fedtn_mps_summary.csv"),
            "p13_evaluation": _relative(REPORTS_DIR / "p13_dashboard_model_evaluation.csv"),
            "model_registry": _relative(DASHBOARD_DIR / "model_registry.json"),
        },
        "warnings": warnings,
    }
    return summary, warnings


def main() -> int:
    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary, warnings = build_dashboard_summary()
    summary_path = DASHBOARD_DATA_DIR / "dashboard_summary.json"
    warnings_path = REPORTS_DIR / "p13_dashboard_warnings.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    warnings_path.write_text(json.dumps({"warnings": warnings}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Dashboard summary written: {summary_path}")
    print(f"Warnings written: {warnings_path}")
    print(f"warnings={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
