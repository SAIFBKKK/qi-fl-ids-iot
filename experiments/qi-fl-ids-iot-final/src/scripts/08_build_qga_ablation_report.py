"""Build the P8 QGA ablation report from official full-run summaries.

FedAvg + QGA must use the true Flower runtime from
`outputs/qga_fedavg_flower_l1`. The older `outputs/qga_fedavg_l1` directory is
kept as an in-process experimental helper and must not be used as the final P8
FedAvg baseline.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qga.config import load_config, load_json, repo_path, write_json
from qga.report_builder import write_csv


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _test_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    test = summary.get("test", {})
    if isinstance(test, dict) and isinstance(test.get("metrics"), dict):
        return test["metrics"]
    return test if isinstance(test, dict) else {}


def _selected_features_count(summary: dict[str, Any]) -> int | None:
    for path in [
        ("qga", "selected_features_count"),
        ("dataset", "input_dim_selected"),
        ("model", "qga_selected_features_count"),
        ("model", "input_dim"),
    ]:
        value: Any = summary
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        parsed = _as_int(value)
        if parsed is not None:
            return parsed
    return None


def is_valid_full_flower_summary(summary: dict[str, Any], *, required_rounds: int) -> bool:
    criteria = summary.get("criteria", {})
    training = summary.get("training", {})
    dataset = summary.get("dataset", {})
    scenario = summary.get("scenario", {})
    rounds_completed = _as_int(training.get("rounds_completed"))
    rounds_configured = _as_int(training.get("rounds_configured"), _as_int(scenario.get("rounds")))
    return bool(
        summary.get("accepted")
        and summary.get("true_flower_runtime") is True
        and criteria.get("true_flower_runtime") is True
        and summary.get("mode") == "full"
        and rounds_completed == int(required_rounds)
        and rounds_configured == int(required_rounds)
        and dataset.get("test_sent_to_clients") is False
        and criteria.get("test_sent_to_clients_false") is True
    )


def is_calibrated_flower_summary(summary: dict[str, Any]) -> bool:
    qga = summary.get("qga", {})
    dataset = summary.get("dataset", {})
    return bool(
        summary.get("calibration_decision_used") is True
        and summary.get("selected_mask_source") == "final_selected_mask"
        and qga.get("calibration_decision_used") is True
        and qga.get("selected_mask_source") == "final_selected_mask"
        and qga.get("selected_mask_id") == "conservative_seed_42"
        and _selected_features_count(summary) == 12
        and dataset.get("input_dim_selected") == 12
    )


def _iter_run_summaries(root: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    summaries: list[tuple[str, Path, dict[str, Any]]] = []
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return summaries
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "artifacts" / "run_summary.json"
        if summary_path.exists():
            summaries.append((run_dir.name, summary_path, load_json(summary_path)))
    return summaries


def find_latest_valid_qga_flower_run(
    config: dict[str, Any],
    *,
    alpha: float = 0.5,
    clients: int = 3,
    rounds: int = 30,
    require_calibrated: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    scenario_dir = (
        repo_path(config, "outputs.qga_fedavg_flower_dir")
        / f"alpha_{alpha}"
        / f"k{clients}"
    )
    warnings: list[str] = []
    candidates = _iter_run_summaries(scenario_dir)
    smoke_runs = [run_id for run_id, _, summary in candidates if summary.get("mode") == "smoke"]
    if smoke_runs:
        warnings.append(
            "Ignored P8 FedAvg+QGA Flower smoke run(s): " + ", ".join(smoke_runs)
        )
    in_process_latest = repo_path(config, "outputs.qga_fedavg_dir") / f"alpha_{alpha}" / f"k{clients}" / "latest_run_summary.json"
    if in_process_latest.exists():
        warnings.append(
            "Ignored in-process P8 FedAvg+QGA helper summary: "
            + in_process_latest.as_posix()
        )
    valid = [
        (run_id, path, summary)
        for run_id, path, summary in candidates
        if is_valid_full_flower_summary(summary, required_rounds=rounds)
    ]
    if require_calibrated:
        non_calibrated = [
            run_id for run_id, _, summary in valid if not is_calibrated_flower_summary(summary)
        ]
        if non_calibrated:
            warnings.append(
                "Ignored non-calibrated P8 FedAvg+QGA Flower full run(s): "
                + ", ".join(non_calibrated)
            )
        valid = [
            (run_id, path, summary)
            for run_id, path, summary in valid
            if is_calibrated_flower_summary(summary)
        ]
    if not valid:
        raise RuntimeError(
            f"No valid full true-Flower P8 FedAvg+QGA run found under {scenario_dir}. "
            "Expected accepted=true, true_flower_runtime=true, mode=full, rounds_completed=30, "
            "test_sent_to_clients=false, calibration_decision_used=true, "
            "selected_mask_source=final_selected_mask, and selected_mask_id=conservative_seed_42."
        )
    run_id, summary_path, summary = sorted(valid, key=lambda item: item[0])[-1]
    summary["_source_summary"] = summary_path.as_posix()
    summary["_selected_run_id"] = run_id
    return summary, warnings


def _p5_baseline_row(config: dict[str, Any]) -> dict[str, Any]:
    p5_summary = repo_path(config, "inputs.p5_grid_summary")
    if not p5_summary.exists():
        raise FileNotFoundError(f"P5 grid summary not found: {p5_summary}")
    with p5_summary.open("r", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["alpha"] == "0.5" and row["clients"] == "3":
                return {
                    "method": "P5 FedAvg baseline",
                    "features_count": 28,
                    "feature_reduction_ratio": 0.0,
                    "macro_f1": _as_float(row["macro_f1"]),
                    "attack_recall": _as_float(row["attack_recall"]),
                    "fpr": _as_float(row["fpr"]),
                    "accuracy": _as_float(row["accuracy"]),
                    "model_size_bytes": _as_int(row["model_size_bytes"]),
                    "bandwidth_total_bytes": _as_int(row["bandwidth_total_bytes"]),
                    "gap_macro_f1_vs_baseline": 0.0,
                    "bandwidth_reduction_ratio": 0.0,
                    "accepted": True,
                    "true_flower_runtime": False,
                    "runtime": "in_process_p5_baseline",
                    "mode": "full",
                    "rounds": _as_int(row["rounds"], 30),
                    "run_id": "",
                    "source_summary": p5_summary.as_posix(),
                    "notes": "P5 in-process scientific FedAvg baseline",
                }
    raise RuntimeError("P5 baseline alpha=0.5 K=3 not found in grid summary")


def _flower_row_from_summary(config: dict[str, Any], summary: dict[str, Any], p5_row: dict[str, Any]) -> dict[str, Any]:
    metrics = _test_metrics(summary)
    comm = summary.get("communication", {})
    model = summary.get("model", {})
    scenario = summary.get("scenario", {})
    features = _selected_features_count(summary)
    model_size = _as_int(comm.get("model_size_bytes"), _as_int(model.get("model_size_bytes")))
    clients = _as_int(scenario.get("clients"), 3) or 3
    rounds = _as_int(summary.get("training", {}).get("rounds_completed"), _as_int(scenario.get("rounds"), 30)) or 30
    formula_bandwidth = 2 * clients * int(model_size or 0) * rounds
    bandwidth = _as_int(comm.get("communication_cumulative_bytes"), formula_bandwidth)
    macro_f1 = _as_float(metrics.get("macro_f1"), 0.0) or 0.0
    return {
        "method": "P8 FedAvg + QGA Flower",
        "features_count": features,
        "feature_reduction_ratio": 1 - float(features) / 28.0 if features else "",
        "macro_f1": macro_f1,
        "attack_recall": _as_float(metrics.get("recall_attack")),
        "fpr": _as_float(metrics.get("FPR")),
        "accuracy": _as_float(metrics.get("accuracy")),
        "model_size_bytes": model_size,
        "bandwidth_total_bytes": bandwidth,
        "gap_macro_f1_vs_baseline": macro_f1 - float(p5_row["macro_f1"]),
        "bandwidth_reduction_ratio": 1 - float(bandwidth or 0) / float(p5_row["bandwidth_total_bytes"]),
        "accepted": summary.get("accepted") is True,
        "true_flower_runtime": True,
        "runtime": summary.get("runtime", "manual"),
        "mode": summary.get("mode"),
        "rounds": rounds,
        "run_id": summary.get("run_id"),
        "selected_mask_id": summary.get("selected_mask_id") or summary.get("qga", {}).get("selected_mask_id"),
        "selected_mask_source": summary.get("selected_mask_source") or summary.get("qga", {}).get("selected_mask_source"),
        "calibration_decision_used": summary.get("calibration_decision_used") is True,
        "source_summary": summary.get("_source_summary", ""),
        "notes": f"Official full true-Flower P8 run; bandwidth formula={formula_bandwidth}",
    }


def _heterofl_row(config: dict[str, Any], p5_row: dict[str, Any]) -> dict[str, Any] | None:
    path = repo_path(config, "outputs.qga_heterofl_dir") / "alpha_0.5" / "k3" / "latest_run_summary.json"
    if not path.exists():
        return None
    summary = load_json(path)
    metrics = _test_metrics(summary)
    comm = summary.get("communication", {})
    features = _selected_features_count(summary)
    macro_f1 = _as_float(metrics.get("macro_f1"), 0.0) or 0.0
    return {
        "method": "P8 HeteroFL + QGA",
        "features_count": features,
        "feature_reduction_ratio": 1 - float(features) / 28.0 if features else "",
        "macro_f1": macro_f1,
        "attack_recall": _as_float(metrics.get("recall_attack")),
        "fpr": _as_float(metrics.get("FPR")),
        "accuracy": _as_float(metrics.get("accuracy")),
        "model_size_bytes": _as_int(comm.get("model_size_bytes")),
        "bandwidth_total_bytes": _as_int(comm.get("total_bytes")),
        "gap_macro_f1_vs_baseline": macro_f1 - float(p5_row["macro_f1"]),
        "bandwidth_reduction_ratio": "",
        "accepted": summary.get("accepted"),
        "true_flower_runtime": False,
        "runtime": "experimental_in_process",
        "mode": summary.get("mode"),
        "rounds": _as_int(summary.get("scenario", {}).get("rounds")),
        "run_id": summary.get("run_id"),
        "source_summary": path.as_posix(),
        "notes": "Experimental in-process HeteroFL+QGA; not final Flower baseline",
    }


def build_ablation_rows(config: dict[str, Any], *, rounds: int = 30) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    p5 = _p5_baseline_row(config)
    flower_summary, flower_warnings = find_latest_valid_qga_flower_run(config, rounds=rounds)
    warnings.extend(flower_warnings)
    rows = [p5, _flower_row_from_summary(config, flower_summary, p5)]
    heterofl = _heterofl_row(config, p5)
    if heterofl is not None:
        rows.append(heterofl)
    return rows, warnings


def _write_markdown(path: Path, rows: list[dict[str, Any]], warnings: list[str]) -> None:
    columns = [
        "method",
        "features_count",
        "macro_f1",
        "attack_recall",
        "fpr",
        "bandwidth_total_bytes",
        "true_flower_runtime",
        "calibration_decision_used",
        "selected_mask_source",
        "accepted",
    ]
    header = "| " + " | ".join(columns) + " |\n"
    header += "|" + "|".join("---" for _ in columns) + "|\n"
    body = ""
    for row in rows:
        body += "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |\n"
    warning_block = ""
    if warnings:
        warning_block = "\n## Warnings\n\n" + "\n".join(f"- {warning}" for warning in warnings) + "\n"
    path.write_text(header + body + warning_block, encoding="utf-8")


def _save_figures(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    figures_dir = repo_path(config, "outputs.figures_dir") / "ablation"
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row["method"]).replace(" + ", "\n+ ") for row in rows]

    def save(fig: plt.Figure, name: str) -> str:
        path = figures_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path.as_posix()

    paths: list[str] = []
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, [_as_float(row.get("macro_f1"), 0.0) or 0.0 for row in rows], color="#2f6fbb")
    ax.set_ylabel("Macro-F1")
    ax.set_title("P8 QGA ablation Macro-F1")
    ax.set_ylim(0, 1)
    paths.append(save(fig, "p8_ablation_table.png"))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, [_as_int(row.get("bandwidth_total_bytes"), 0) or 0 for row in rows], color="#8a5fbf")
    ax.set_ylabel("Total bytes")
    ax.set_title("P8 QGA bandwidth")
    paths.append(save(fig, "p8_bandwidth_reduction.png"))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, [_as_int(row.get("model_size_bytes"), 0) or 0 for row in rows], color="#3b8f5a")
    ax.set_ylabel("Model size bytes")
    ax.set_title("P8 QGA model size")
    paths.append(save(fig, "p8_model_size_reduction.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        [_as_int(row.get("features_count"), 0) or 0 for row in rows],
        [_as_float(row.get("macro_f1"), 0.0) or 0.0 for row in rows],
        s=90,
        color="#d95f02",
    )
    for row in rows:
        ax.annotate(str(row["method"]).split(" baseline")[0], (_as_int(row.get("features_count"), 0) or 0, _as_float(row.get("macro_f1"), 0.0) or 0.0), fontsize=8)
    ax.set_xlabel("Features")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 vs selected features")
    ax.grid(alpha=0.3)
    paths.append(save(fig, "p8_macro_f1_vs_features.png"))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rounds", type=int, default=30)
    args = parser.parse_args()
    config = load_config(args.config)
    reports_dir = repo_path(config, "outputs.reports_dir")
    rows, warnings = build_ablation_rows(config, rounds=int(args.rounds))

    csv_path = reports_dir / "p8_qga_ablation_summary.csv"
    json_path = reports_dir / "p8_qga_ablation_summary.json"
    md_path = reports_dir / "p8_qga_ablation_table.md"
    warnings_path = reports_dir / "p8_qga_ablation_warnings.json"
    figures_manifest = reports_dir / "p8_qga_ablation_figures_manifest.json"

    write_csv(csv_path, rows)
    write_json(json_path, rows)
    write_json(warnings_path, warnings)
    _write_markdown(md_path, rows, warnings)
    figures = _save_figures(config, rows)
    write_json(figures_manifest, figures)

    print(f"P8 ablation report built: {csv_path}")
    if warnings:
        print("Warnings:", "; ".join(warnings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
