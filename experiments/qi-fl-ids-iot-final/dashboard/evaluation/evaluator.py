from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .artifact_resolver import first_existing, repo_root, resolve
from .metrics import binary_metrics
from .model_registry import load_registry, load_reported_rows
from .test_loader import apply_mask, load_qga_indices, load_test_npz


def _import_model_class():
    src = repo_root() / "experiments/qi-fl-ids-iot-final/src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from models.l1_mlp import CentralizedL1MLP

    return CentralizedL1MLP


def _state_dict_from_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        if all(hasattr(value, "shape") for value in payload.values()):
            return payload
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _evaluate_checkpoint(model_info: dict[str, Any], checkpoint: Path, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    import torch

    Model = _import_model_class()
    model = Model(input_dim=int(model_info["feature_count"]), hidden_layers=[128, 64], output_dim=2, dropout=0.2)
    state = _state_dict_from_checkpoint(checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(y), 4096):
            batch = torch.as_tensor(X[start : start + 4096], dtype=torch.float32)
            logits = model(batch)
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    return binary_metrics(y, np.concatenate(predictions))


def _reported_metrics(model_id: str, reported_rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    row = reported_rows.get(model_id, {})
    return {
        "macro_f1": row.get("macro_f1", ""),
        "weighted_f1": row.get("weighted_f1", ""),
        "attack_recall": row.get("attack_recall", ""),
        "fpr": row.get("fpr", ""),
        "accuracy": row.get("accuracy", ""),
        "model_size_bytes": row.get("model_size_bytes", ""),
        "bandwidth_total_bytes": row.get("bandwidth_total_bytes", ""),
    }


def evaluate_models(registry_path: str | Path | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    registry = load_registry(registry_path)
    reported_rows = load_reported_rows("experiments/qi-fl-ids-iot-final/outputs/reports/p12_global_ablation_summary.csv")
    warnings: list[str] = []
    X: np.ndarray | None = None
    y: np.ndarray | None = None
    test_path = registry.get("test_npz")
    if test_path and resolve(test_path).exists():
        X, y = load_test_npz(test_path)
    else:
        warnings.append("test_scaled_npz_unavailable_report_only_mode")

    qga_indices = None
    if registry.get("qga_mask_source") and resolve(registry["qga_mask_source"]).exists():
        qga_indices = load_qga_indices(registry["qga_mask_source"])

    rows: list[dict[str, Any]] = []
    for model in registry["models"]:
        checkpoint = first_existing(model.get("checkpoint_candidates", []))
        status = "report_only"
        metrics = _reported_metrics(model["model_id"], reported_rows)
        warning = ""
        if checkpoint and X is not None and y is not None:
            try:
                X_eval = apply_mask(X, qga_indices if model.get("uses_qga_mask") else None)
                metrics = {**metrics, **_evaluate_checkpoint(model, checkpoint, X_eval, y)}
                status = "evaluable"
            except Exception as exc:  # noqa: BLE001 - dashboard must stay available
                warning = f"checkpoint_evaluation_failed: {exc}"
                warnings.append(f"{model['model_id']}: {warning}")
                status = "report_only"
        else:
            missing = "checkpoint" if not checkpoint else "test_npz"
            warning = f"model artifact unavailable ({missing}); using reported metrics"
            warnings.append(f"{model['model_id']}: {warning}")
        rows.append(
            {
                "model_id": model["model_id"],
                "phase": model["phase"],
                "method": model["method"],
                "task": model["task"],
                "feature_count": model["feature_count"],
                "uses_qga_mask": model["uses_qga_mask"],
                "selected_mask_id": model.get("selected_mask_id") or "",
                "status": status,
                "recommended_use_case": model["recommended_use_case"],
                "checkpoint": str(checkpoint) if checkpoint else "",
                "metric_source": "recomputed_test" if status == "evaluable" else "reported",
                "macro_f1": metrics.get("macro_f1", ""),
                "weighted_f1": metrics.get("weighted_f1", ""),
                "attack_recall": metrics.get("attack_recall", ""),
                "fpr": metrics.get("fpr", ""),
                "fnr": metrics.get("fnr", ""),
                "accuracy": metrics.get("accuracy", ""),
                "TP": metrics.get("TP", ""),
                "TN": metrics.get("TN", ""),
                "FP": metrics.get("FP", ""),
                "FN": metrics.get("FN", ""),
                "confusion_matrix": metrics.get("confusion_matrix", ""),
                "model_size_bytes": metrics.get("model_size_bytes", ""),
                "bandwidth_total_bytes": metrics.get("bandwidth_total_bytes", ""),
                "warning": warning,
            }
        )
    return rows, warnings


def write_evaluation_outputs(rows: list[dict[str, Any]], warnings: list[str]) -> None:
    reports = repo_root() / "experiments/qi-fl-ids-iot-final/outputs/reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "p13_dashboard_model_evaluation.csv"
    json_path = reports / "p13_dashboard_model_evaluation.json"
    md_path = reports / "p13_dashboard_model_evaluation_table.md"
    fields = [
        "model_id",
        "phase",
        "method",
        "feature_count",
        "status",
        "metric_source",
        "macro_f1",
        "weighted_f1",
        "attack_recall",
        "fpr",
        "fnr",
        "accuracy",
        "TP",
        "TN",
        "FP",
        "FN",
        "model_size_bytes",
        "bandwidth_total_bytes",
        "warning",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    json_path.write_text(json.dumps({"models": rows, "warnings": warnings}, indent=2, sort_keys=True), encoding="utf-8")
    header = "| model | status | source | macro_f1 | attack_recall | fpr | accuracy |\n|---|---|---|---:|---:|---:|---:|\n"
    lines = [
        f"| {row['method']} | {row['status']} | {row['metric_source']} | {row.get('macro_f1', '')} | {row.get('attack_recall', '')} | {row.get('fpr', '')} | {row.get('accuracy', '')} |"
        for row in rows
    ]
    md_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
