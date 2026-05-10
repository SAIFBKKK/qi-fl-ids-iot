"""Run P8-b QGA L2 Flower short validation for candidate masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga_l2.config import load_config, repo_path, write_json
from qga_l2.flower_runtime import run_smoke_subprocess
from qga_l2.fitness_l2 import macro_metrics_from_confusion


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        parsed = _float_or_none(metrics.get(key))
        if parsed is not None:
            return parsed
    return None


def _derived_confusion_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    matrix = metrics.get("confusion_matrix")
    if not matrix:
        return {}
    return macro_metrics_from_confusion(np.asarray(matrix, dtype=int))


def _parse_mask_id(mask_id: str) -> tuple[str, str]:
    marker = "_seed_"
    if marker not in mask_id:
        return mask_id, ""
    profile, seed = mask_id.rsplit(marker, 1)
    return profile, seed


def _row_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("validation", {}).get("metrics", {})
    derived = _derived_confusion_metrics(metrics)
    mask_id = str(summary.get("selected_mask_id", ""))
    profile, seed = _parse_mask_id(mask_id)
    scenario = summary.get("scenario", {})
    training = summary.get("training", {})
    dataset = summary.get("dataset", {})
    qga = summary.get("qga", {})
    communication = summary.get("communication", {})
    model = summary.get("model", {})
    rounds = training.get("rounds_completed", scenario.get("rounds", ""))
    row = {
        "mask_id": mask_id,
        "profile": profile,
        "seed": seed,
        "alpha": scenario.get("alpha", ""),
        "clients": scenario.get("clients", ""),
        "rounds": rounds,
        "features_count": qga.get("selected_features_count", dataset.get("input_dim_selected", "")),
        "val_macro_f1": _metric(metrics, "macro_f1") if _metric(metrics, "macro_f1") is not None else derived.get("macro_f1", ""),
        "val_macro_recall": _metric(metrics, "macro_recall", "recall_macro")
        if _metric(metrics, "macro_recall", "recall_macro") is not None
        else derived.get("macro_recall", ""),
        "val_macro_precision": _metric(metrics, "macro_precision", "precision_macro")
        if _metric(metrics, "macro_precision", "precision_macro") is not None
        else derived.get("macro_precision", ""),
        "val_macro_fpr": _metric(metrics, "macro_fpr", "FPR_macro")
        if _metric(metrics, "macro_fpr", "FPR_macro") is not None
        else derived.get("macro_fpr", ""),
        "val_weighted_f1": _metric(metrics, "weighted_f1") if _metric(metrics, "weighted_f1") is not None else "",
        "bandwidth_total_bytes": communication.get("communication_cumulative_bytes", communication.get("total_bytes", "")),
        "model_size_bytes": model.get("model_size_bytes", ""),
        "true_flower_runtime": summary.get("true_flower_runtime", summary.get("criteria", {}).get("true_flower_runtime", "")),
        "test_sent_to_clients": dataset.get("test_sent_to_clients", not summary.get("criteria", {}).get("test_sent_to_clients_false", False)),
        "accepted": summary.get("accepted", ""),
        "run_id": summary.get("run_id", ""),
    }
    return row


def _enrich_row_from_artifacts(row: dict[str, Any], artifacts_dir: Path) -> dict[str, Any]:
    model_config = artifacts_dir / "model_config.json"
    if model_config.exists():
        with model_config.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        row["model_size_bytes"] = payload.get("model_size_bytes", row.get("model_size_bytes", ""))
        row["features_count"] = payload.get("selected_features_count", row.get("features_count", ""))
    bandwidth = artifacts_dir / "bandwidth_rounds.csv"
    if bandwidth.exists() and bandwidth.stat().st_size:
        rows = _read_rows(bandwidth)
        if rows:
            last = rows[-1]
            row["bandwidth_total_bytes"] = last.get("communication_cumulative_bytes", row.get("bandwidth_total_bytes", ""))
            row["model_size_bytes"] = last.get("model_size_bytes", row.get("model_size_bytes", ""))
    return row


def _rebuild_from_existing(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = repo_path(config, "outputs.qga_l2_flower_dir")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("alpha_*/k*/runs/*/artifacts/run_summary.json")):
        with path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        row = _enrich_row_from_artifacts(_row_from_summary(summary), path.parent)
        if int(float(row.get("rounds") or 0)) == 5:
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-samples-per-client", type=int, default=1000)
    parser.add_argument("--address", default=None)
    parser.add_argument("--only-first", action="store_true")
    parser.add_argument("--rebuild-from-existing", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    reports = repo_path(config, "outputs.reports_dir")
    if args.rebuild_from_existing:
        rows = _rebuild_from_existing(config)
        csv_path = reports / "p8b_qga_l2_flower_short_validation.csv"
        _write_rows(csv_path, rows)
        write_json(reports / "p8b_qga_l2_flower_short_validation.json", rows)
        print(f"P8-b QGA L2 Flower short validation rebuilt from existing summaries | rows={len(rows)}")
        return 0
    sweep_rows = sorted(_read_rows(reports / "p8b_qga_l2_profile_sweep_summary.csv"), key=lambda row: float(row["fitness"]), reverse=True)
    scenarios = config["calibration"]["short_validation_scenarios"]
    output_rows: list[dict] = []
    for candidate in sweep_rows[: int(args.top_n)]:
        for scenario in scenarios:
            summary = run_smoke_subprocess(
                config_path=args.config,
                alpha=float(scenario["alpha"]),
                clients=int(scenario["clients"]),
                rounds=int(args.rounds),
                max_samples_per_client=int(args.max_samples_per_client),
                address=args.address or config["flower"]["address"],
                mask_path=repo_path(config) / candidate["feature_mask_path"],
                evaluate_test=False,
                timeout_sec=900,
            )
            output_rows.append(_row_from_summary(summary))
            if args.only_first:
                break
        if args.only_first:
            break
    csv_path = reports / "p8b_qga_l2_flower_short_validation.csv"
    existing = _read_rows(csv_path) if csv_path.exists() and csv_path.stat().st_size else []
    rows = existing + output_rows
    _write_rows(csv_path, rows)
    write_json(reports / "p8b_qga_l2_flower_short_validation.json", rows)
    print(f"P8-b QGA L2 Flower short validation completed | new_runs={len(output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
