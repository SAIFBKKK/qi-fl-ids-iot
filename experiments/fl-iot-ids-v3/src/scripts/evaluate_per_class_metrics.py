from __future__ import annotations

import csv
import json
import math
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.common.paths import ARTIFACTS_DIR, DATA_DIR, OUTPUTS_DIR
from src.model.network import MLPClassifier


NUM_CLASSES = 34
LABELS = list(range(NUM_CLASSES))
RARE_CLASS_IDS = (0, 3, 30, 31, 33)
REPORT_DIR = OUTPUTS_DIR / "reports" / "qi_benchmark_reduced" / "per_class_metrics"
BASELINE_DIR = OUTPUTS_DIR / "reports" / "baselines"


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    experiment_name: str
    scenario: str
    features: str
    uses_qga: bool


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec("E1", "exp_bench30_normal_fedavg_28f", "normal_noniid", "28f", False),
    ExperimentSpec("E2", "exp_bench30_normal_qifa_28f", "normal_noniid", "28f", False),
    ExperimentSpec("E3", "exp_bench30_normal_fedavg_qga15", "normal_noniid", "QGA-15", True),
    ExperimentSpec("E4", "exp_bench30_normal_qifa_qga15", "normal_noniid", "QGA-15", True),
    ExperimentSpec("E5", "exp_bench30_absent_fedavg_28f", "absent_local", "28f", False),
    ExperimentSpec("E6", "exp_bench30_absent_qifa_28f", "absent_local", "28f", False),
    ExperimentSpec("E7", "exp_bench30_absent_fedavg_qga15", "absent_local", "QGA-15", True),
    ExperimentSpec("E8", "exp_bench30_absent_qifa_qga15", "absent_local", "QGA-15", True),
)

QIFA_COMPARISONS = (
    ("E2 - E1", "normal_noniid", "28f", "E2", "E1"),
    ("E6 - E5", "absent_local", "28f", "E6", "E5"),
    ("E4 - E3", "normal_noniid", "QGA-15", "E4", "E3"),
    ("E8 - E7", "absent_local", "QGA-15", "E8", "E7"),
)

QGA_COMPARISONS = (
    ("E3 - E1", "normal_noniid", "FedAvg: QGA-15 - 28f", "E3", "E1"),
    ("E4 - E2", "normal_noniid", "QIFA: QGA-15 - 28f", "E4", "E2"),
    ("E7 - E5", "absent_local", "FedAvg: QGA-15 - 28f", "E7", "E5"),
    ("E8 - E6", "absent_local", "QIFA: QGA-15 - 28f", "E8", "E6"),
)


def nan_if_zero_denominator(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def compute_per_class_metrics(
    cm: np.ndarray,
    class_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute one-vs-rest metrics from a square confusion matrix."""

    matrix = np.asarray(cm)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square confusion matrix, got shape={matrix.shape}.")

    n_classes = matrix.shape[0]
    names = class_names or [f"class_{idx}" for idx in range(n_classes)]
    if len(names) != n_classes:
        raise ValueError(f"Expected {n_classes} class names, got {len(names)}.")

    total = int(matrix.sum())
    rows: list[dict[str, Any]] = []
    for class_id in range(n_classes):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum() - tp)
        fn = int(matrix[class_id, :].sum() - tp)
        tn = int(total - tp - fp - fn)

        accuracy = nan_if_zero_denominator(tp + tn, total)
        precision = nan_if_zero_denominator(tp, tp + fp)
        recall = nan_if_zero_denominator(tp, tp + fn)
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if precision > 0.0 and recall > 0.0
            else float("nan")
        )
        specificity = nan_if_zero_denominator(tn, tn + fp)
        fpr = nan_if_zero_denominator(fp, fp + tn)
        fnr = nan_if_zero_denominator(fn, fn + tp)
        support = int(matrix[class_id, :].sum())

        rows.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "specificity": specificity,
                "fpr": fpr,
                "fnr": fnr,
                "support": support,
            }
        )
    return rows


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value


def _write_dict_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in matrix:
            writer.writerow([_csv_value(float(value)) if np.issubdtype(matrix.dtype, np.floating) else int(value) for value in row])


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_class_names() -> list[str]:
    candidates = [
        ARTIFACTS_DIR / "baseline" / "artifacts" / "label_mapping_34.pkl",
        ARTIFACTS_DIR / "label_mapping_34.pkl",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        id_to_label = payload.get("id_to_label") if isinstance(payload, Mapping) else None
        if isinstance(id_to_label, Mapping):
            return [str(id_to_label.get(idx, id_to_label.get(str(idx), f"class_{idx}"))) for idx in LABELS]
    return [f"class_{idx}" for idx in LABELS]


def _experiment_dir(experiment_name: str) -> Path:
    candidates = [
        OUTPUTS_DIR / experiment_name,
        BASELINE_DIR / experiment_name,
    ]
    for path in candidates:
        if (path / "best_checkpoint.pth").exists():
            return path
    raise FileNotFoundError(
        f"Could not find best_checkpoint.pth for {experiment_name}. "
        f"Tried: {', '.join(str(path) for path in candidates)}"
    )


def load_trusted_local_checkpoint(checkpoint_path: Path) -> Any:
    try:
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")

    for key in ("model_state_dict", "state_dict", "model"):
        if key not in checkpoint:
            continue
        candidate = checkpoint[key]
        if isinstance(candidate, torch.nn.Module):
            return candidate.state_dict()
        if isinstance(candidate, Mapping):
            return {str(k).removeprefix("module."): v for k, v in candidate.items()}
        raise TypeError(f"Checkpoint key {key!r} has unsupported type: {type(candidate).__name__}")

    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return {str(k).removeprefix("module."): v for k, v in checkpoint.items()}
    raise KeyError("Checkpoint does not contain a supported model state dict.")


def _build_model(config: dict[str, Any], input_dim: int) -> MLPClassifier:
    model_cfg = dict(config.get("model", {}))
    return MLPClassifier(
        input_dim=input_dim,
        num_classes=int(model_cfg.get("output_dim", config.get("dataset", {}).get("num_classes", NUM_CLASSES))),
        hidden_dims=tuple(model_cfg.get("hidden_dims", [256, 128])),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )


def _load_qga_mask(scenario: str) -> np.ndarray:
    path = ARTIFACTS_DIR / "qi_feature_selection" / scenario / "feature_mask.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    mask = np.load(path)
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if mask.shape != (28,):
        raise ValueError(f"Expected QGA mask shape (28,), got {mask.shape} from {path}.")
    if int(mask.sum()) != 15:
        raise ValueError(f"Expected QGA mask with 15 selected features, got {int(mask.sum())}.")
    return mask


def _load_validation_data(scenario: str) -> tuple[np.ndarray, np.ndarray, str]:
    global_path = DATA_DIR / "processed" / scenario / "global_val.npz"
    if global_path.exists():
        data = np.load(global_path, allow_pickle=True)
        return np.asarray(data["X"], dtype=np.float32), np.asarray(data["y"], dtype=np.int64), str(global_path)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    source_parts: list[str] = []
    for node_id in ("node1", "node2", "node3"):
        path = DATA_DIR / "processed" / scenario / node_id / "val_preprocessed.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True)
        x_parts.append(np.asarray(data["X"], dtype=np.float32))
        y_parts.append(np.asarray(data["y"], dtype=np.int64))
        source_parts.append(str(path))
    return np.vstack(x_parts), np.concatenate(y_parts), "concatenated:" + "|".join(source_parts)


def _load_model_for_experiment(spec: ExperimentSpec, input_dim: int) -> tuple[MLPClassifier, dict[str, Any], Path]:
    run_dir = _experiment_dir(spec.experiment_name)
    resolved = _read_json(run_dir / "resolved_config.json")
    config = dict(resolved["config"]) if resolved is not None and "config" in resolved else {}
    if config:
        model_input_dim = int(config.get("model", {}).get("input_dim", input_dim))
        if model_input_dim != input_dim:
            raise ValueError(
                f"{spec.experiment_id} input dim mismatch: config says {model_input_dim}, "
                f"evaluation data has {input_dim}."
            )
    model = _build_model(config, input_dim)
    checkpoint_path = run_dir / "best_checkpoint.pth"
    checkpoint = load_trusted_local_checkpoint(checkpoint_path)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, config, checkpoint_path


def _predict(model: MLPClassifier, x: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = torch.from_numpy(np.asarray(x[start : start + batch_size], dtype=np.float32))
            logits = model(batch)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds)


def _classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    return classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=class_names,
        output_dict=True,
        zero_division=np.nan,
    )


def _finite_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [row for row in rows if isinstance(row.get(key), float) and np.isfinite(row[key])]


def _format_float(value: Any, digits: int = 4) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, (float, int, np.floating, np.integer)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _write_per_experiment_summary(
    out_dir: Path,
    spec: ExperimentSpec,
    rows: list[dict[str, Any]],
) -> None:
    finite_f1 = _finite_rows(rows, "f1_score")
    worst = sorted(finite_f1, key=lambda row: row["f1_score"])[:5]
    best = sorted(finite_f1, key=lambda row: row["f1_score"], reverse=True)[:5]
    rare = [row for row in rows if int(row["support"]) < 1000]

    lines = [
        f"# {spec.experiment_id} per-class summary",
        "",
        f"- Experiment: `{spec.experiment_name}`",
        f"- Scenario: `{spec.scenario}`",
        f"- Features: `{spec.features}`",
        "",
        "## Top 5 worst F1 classes",
        "",
    ]
    for row in worst:
        lines.append(f"- {row['class_id']} {row['class_name']}: F1={_format_float(row['f1_score'])}, support={row['support']}")
    lines.extend(["", "## Top 5 best F1 classes", ""])
    for row in best:
        lines.append(f"- {row['class_id']} {row['class_name']}: F1={_format_float(row['f1_score'])}, support={row['support']}")
    lines.extend(["", "## Rare classes summary", ""])
    if rare:
        for row in sorted(rare, key=lambda item: (int(item["support"]), int(item["class_id"]))):
            lines.append(
                f"- {row['class_id']} {row['class_name']}: support={row['support']}, "
                f"recall={_format_float(row['recall'])}, F1={_format_float(row['f1_score'])}"
            )
    else:
        lines.append("- No class has support below 1000 in this evaluation set.")
    lines.extend(
        [
            "",
            "## Brief interpretation",
            "",
            "Low-F1 rows should be read together with support. Undefined values are written as NaN when a denominator is zero, so absent classes are not converted into artificial zeros.",
            "",
        ]
    )
    (out_dir / "per_class_summary.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_experiment(
    spec: ExperimentSpec,
    class_names: list[str],
    scenario_cache: dict[str, tuple[np.ndarray, np.ndarray, str]],
) -> dict[str, Any]:
    if spec.scenario not in scenario_cache:
        scenario_cache[spec.scenario] = _load_validation_data(spec.scenario)
    x_full, y_true, data_source = scenario_cache[spec.scenario]

    if spec.uses_qga:
        mask = _load_qga_mask(spec.scenario)
        x_eval = x_full[:, mask]
    else:
        x_eval = x_full

    model, config, checkpoint_path = _load_model_for_experiment(spec, input_dim=x_eval.shape[1])
    y_pred = _predict(model, x_eval)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_normalized = confusion_matrix(y_true, y_pred, labels=LABELS, normalize="true")
    per_class_rows = compute_per_class_metrics(cm, class_names)
    report = _classification_report(y_true, y_pred, class_names)

    out_dir = REPORT_DIR / spec.experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_matrix_csv(out_dir / "confusion_matrix.csv", cm)
    _write_matrix_csv(out_dir / "confusion_matrix_normalized.csv", cm_normalized)
    per_class_fields = [
        "class_id",
        "class_name",
        "TP",
        "FP",
        "FN",
        "TN",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "specificity",
        "fpr",
        "fnr",
        "support",
    ]
    _write_dict_csv(out_dir / "per_class_metrics.csv", per_class_rows, per_class_fields)
    (out_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    _write_per_experiment_summary(out_dir, spec, per_class_rows)

    return {
        "spec": spec,
        "config": config,
        "checkpoint_path": str(checkpoint_path),
        "data_source": data_source,
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_normalized,
        "per_class_rows": per_class_rows,
    }


def _index_rows(results: dict[str, dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    return {
        experiment_id: {int(row["class_id"]): row for row in result["per_class_rows"]}
        for experiment_id, result in results.items()
    }


def _delta(left: Any, right: Any) -> float:
    if not isinstance(left, (float, int, np.floating, np.integer)):
        return float("nan")
    if not isinstance(right, (float, int, np.floating, np.integer)):
        return float("nan")
    if math.isnan(float(left)) or math.isnan(float(right)):
        return float("nan")
    return float(left) - float(right)


def _interpret_delta_f1(delta_f1: float) -> str:
    if math.isnan(delta_f1):
        return "undefined"
    if delta_f1 > 0.01:
        return "improved"
    if delta_f1 < -0.01:
        return "degraded"
    return "no_change"


def _build_long_comparison(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in EXPERIMENTS:
        for row in results[spec.experiment_id]["per_class_rows"]:
            payload = {
                "experiment_id": spec.experiment_id,
                "experiment_name": spec.experiment_name,
                "scenario": spec.scenario,
                "features": spec.features,
            }
            payload.update(row)
            rows.append(payload)
    return rows


def _build_delta_table(
    results: dict[str, dict[str, Any]],
    comparisons: tuple[tuple[str, str, str, str, str], ...],
) -> list[dict[str, Any]]:
    indexed = _index_rows(results)
    rows: list[dict[str, Any]] = []
    for comparison, scenario, features, left_id, right_id in comparisons:
        for class_id in LABELS:
            left = indexed[left_id][class_id]
            right = indexed[right_id][class_id]
            delta_precision = _delta(left["precision"], right["precision"])
            delta_recall = _delta(left["recall"], right["recall"])
            delta_f1 = _delta(left["f1_score"], right["f1_score"])
            delta_fpr = _delta(left["fpr"], right["fpr"])
            rows.append(
                {
                    "comparison": comparison,
                    "scenario": scenario,
                    "features": features,
                    "class_id": class_id,
                    "class_name": left["class_name"],
                    "delta_precision": delta_precision,
                    "delta_recall": delta_recall,
                    "delta_f1": delta_f1,
                    "delta_fpr": delta_fpr,
                    "interpretation": _interpret_delta_f1(delta_f1),
                }
            )
    return rows


def _write_cross_experiment_tables(results: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows = _build_long_comparison(results)
    long_fields = [
        "experiment_id",
        "experiment_name",
        "scenario",
        "features",
        "class_id",
        "class_name",
        "TP",
        "FP",
        "FN",
        "TN",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "specificity",
        "fpr",
        "fnr",
        "support",
    ]
    _write_dict_csv(REPORT_DIR / "per_class_comparison_all_experiments.csv", long_rows, long_fields)

    qifa_rows = _build_delta_table(results, QIFA_COMPARISONS)
    delta_fields = [
        "comparison",
        "scenario",
        "features",
        "class_id",
        "class_name",
        "delta_precision",
        "delta_recall",
        "delta_f1",
        "delta_fpr",
        "interpretation",
    ]
    _write_dict_csv(REPORT_DIR / "qifa_delta_per_class.csv", qifa_rows, delta_fields)

    qga_rows = _build_delta_table(results, QGA_COMPARISONS)
    _write_dict_csv(REPORT_DIR / "qga_delta_per_class.csv", qga_rows, delta_fields)
    return long_rows, qifa_rows, qga_rows


def _metric_matrix(results: dict[str, dict[str, Any]], key: str) -> np.ndarray:
    matrix = np.full((len(EXPERIMENTS), NUM_CLASSES), np.nan, dtype=float)
    for row_idx, spec in enumerate(EXPERIMENTS):
        rows = results[spec.experiment_id]["per_class_rows"]
        for row in rows:
            matrix[row_idx, int(row["class_id"])] = float(row[key])
    return matrix


def _mean_support_by_class(results: dict[str, dict[str, Any]]) -> np.ndarray:
    support = np.zeros((len(EXPERIMENTS), NUM_CLASSES), dtype=float)
    for row_idx, spec in enumerate(EXPERIMENTS):
        for row in results[spec.experiment_id]["per_class_rows"]:
            support[row_idx, int(row["class_id"])] = float(row["support"])
    return support.mean(axis=0)


def _plot_metric_heatmap(
    results: dict[str, dict[str, Any]],
    key: str,
    title: str,
    output_path: Path,
) -> None:
    metric = _metric_matrix(results, key)
    mean_support = _mean_support_by_class(results)
    log_support = np.log10(np.maximum(mean_support, 1.0)).reshape(NUM_CLASSES, 1)

    fig = plt.figure(figsize=(15, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[34, 2.2])
    ax = fig.add_subplot(grid[0, 0])
    support_ax = fig.add_subplot(grid[0, 1])

    image = ax.imshow(metric, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Experiment")
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels([str(idx) for idx in LABELS], rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(EXPERIMENTS)))
    ax.set_yticklabels([spec.experiment_id for spec in EXPERIMENTS])
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01, label=key)

    support_image = support_ax.imshow(log_support, aspect="auto", cmap="Greys")
    support_ax.set_title("log support", fontsize=9)
    support_ax.set_xticks([])
    support_ax.set_yticks(np.arange(0, NUM_CLASSES, 2))
    support_ax.set_yticklabels([str(idx) for idx in range(0, NUM_CLASSES, 2)], fontsize=7)
    support_ax.set_ylabel("Class ID")
    fig.colorbar(support_image, ax=support_ax, fraction=0.25, pad=0.02)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _rows_with_finite_delta(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [row for row in rows if isinstance(row.get(key), float) and np.isfinite(row[key])]


def _plot_delta_panels(
    rows: list[dict[str, Any]],
    key: str,
    title: str,
    output_path: Path,
) -> None:
    comparisons = list(dict.fromkeys(str(row["comparison"]) for row in rows))
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, comparison in zip(axes_flat, comparisons):
        subset = [row for row in rows if row["comparison"] == comparison]
        values = np.asarray([float(row[key]) if isinstance(row[key], float) else np.nan for row in subset])
        y = np.arange(len(subset))
        colors = ["#2ca25f" if np.isfinite(value) and value >= 0 else "#de2d26" for value in values]
        ax.barh(y, np.nan_to_num(values, nan=0.0), color=colors)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(comparison)
        ax.set_yticks(y)
        ax.set_yticklabels([str(row["class_id"]) for row in subset], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(key)
        ax.set_ylabel("Class ID")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _short_name(name: str, max_len: int = 24) -> str:
    return name if len(name) <= max_len else name[: max_len - 1] + "."


def _plot_top10_delta(
    rows: list[dict[str, Any]],
    key: str,
    largest: bool,
    title: str,
    output_path: Path,
) -> None:
    finite = _rows_with_finite_delta(rows, key)
    selected = sorted(finite, key=lambda row: float(row[key]), reverse=largest)[:10]
    selected = list(reversed(selected))
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    values = [float(row[key]) for row in selected]
    labels = [
        f"{row['comparison']} | c{row['class_id']} {_short_name(str(row['class_name']))}"
        for row in selected
    ]
    colors = ["#2ca25f" if value >= 0 else "#de2d26" for value in values]
    y = np.arange(len(selected))
    ax.barh(y, values, color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(key)
    ax.set_title(title)
    for idx, (value, row) in enumerate(zip(values, selected)):
        offset = 0.005 if value >= 0 else -0.005
        ha = "left" if value >= 0 else "right"
        ax.text(value + offset, idx, f"c{row['class_id']} {row['class_name']}", va="center", ha=ha, fontsize=7)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_qifa_absent_confusion_delta(results: dict[str, dict[str, Any]], output_path: Path) -> None:
    e5 = np.asarray(results["E5"]["confusion_matrix_normalized"], dtype=float)
    e6 = np.asarray(results["E6"]["confusion_matrix_normalized"], dtype=float)
    raw_delta = e6 - e5
    semantic_delta = raw_delta.copy()
    off_diag = ~np.eye(NUM_CLASSES, dtype=bool)
    semantic_delta[off_diag] *= -1.0
    vmax = max(float(np.nanmax(np.abs(semantic_delta))), 1e-6)

    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    image = ax.imshow(semantic_delta, cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_title("QIFA absent-local confusion delta (E6 vs E5)")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ticks = np.arange(0, NUM_CLASSES, 4)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])
    ax.set_yticklabels([str(tick) for tick in ticks])
    ax.text(
        0.0,
        -0.08,
        "Blue means QIFA moves the row-normalized confusion mass in a better direction: diagonal up, off-diagonal down.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="row-normalized directional delta")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_figures(
    results: dict[str, dict[str, Any]],
    qifa_rows: list[dict[str, Any]],
    qga_rows: list[dict[str, Any]],
) -> None:
    figure_dir = REPORT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_metric_heatmap(
        results,
        "f1_score",
        "Per-class F1 across E1-E8",
        figure_dir / "figure_per_class_f1_heatmap.png",
    )
    _plot_metric_heatmap(
        results,
        "recall",
        "Per-class recall across E1-E8",
        figure_dir / "figure_per_class_recall_heatmap.png",
    )
    _plot_delta_panels(
        qifa_rows,
        "delta_recall",
        "QIFA - FedAvg per-class recall deltas",
        figure_dir / "figure_qifa_delta_recall_per_class.png",
    )
    _plot_delta_panels(
        qga_rows,
        "delta_f1",
        "QGA-15 - 28f per-class F1 deltas",
        figure_dir / "figure_qga_delta_f1_per_class.png",
    )
    _plot_top10_delta(
        qifa_rows,
        "delta_f1",
        True,
        "Top 10 QIFA-improved classes by delta F1",
        figure_dir / "figure_top10_qifa_improved_classes.png",
    )
    _plot_top10_delta(
        qifa_rows,
        "delta_f1",
        False,
        "Top 10 QIFA-degraded classes by delta F1",
        figure_dir / "figure_top10_qifa_degraded_classes.png",
    )
    _plot_top10_delta(
        qga_rows,
        "delta_f1",
        False,
        "Top 10 QGA-degraded classes by delta F1",
        figure_dir / "figure_top10_qga_degraded_classes.png",
    )
    _plot_qifa_absent_confusion_delta(
        results,
        figure_dir / "figure_confusion_matrix_delta_qifa_absent.png",
    )


def _nanmean(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _mean_f1_by_class(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = _index_rows(results)
    rows: list[dict[str, Any]] = []
    for class_id in LABELS:
        values = [float(indexed[spec.experiment_id][class_id]["f1_score"]) for spec in EXPERIMENTS]
        class_name = indexed["E1"][class_id]["class_name"]
        rows.append({"class_id": class_id, "class_name": class_name, "mean_f1": _nanmean(values)})
    return rows


def _rare_by_scenario(results: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    scenario_refs = {"normal_noniid": "E1", "absent_local": "E5"}
    rare: dict[str, list[dict[str, Any]]] = {}
    for scenario, experiment_id in scenario_refs.items():
        rows = results[experiment_id]["per_class_rows"]
        rare[scenario] = [
            row for row in sorted(rows, key=lambda item: (int(item["support"]), int(item["class_id"])))
            if int(row["support"]) < 1000
        ]
    return rare


def _top_abs(rows: list[dict[str, Any]], key: str, count: int = 5) -> list[dict[str, Any]]:
    finite = _rows_with_finite_delta(rows, key)
    return sorted(finite, key=lambda row: abs(float(row[key])), reverse=True)[:count]


def _write_analysis_report(
    results: dict[str, dict[str, Any]],
    qifa_rows: list[dict[str, Any]],
    qga_rows: list[dict[str, Any]],
) -> None:
    weakest = sorted(
        [row for row in _mean_f1_by_class(results) if np.isfinite(row["mean_f1"])],
        key=lambda row: float(row["mean_f1"]),
    )[:5]
    rare = _rare_by_scenario(results)
    qifa_striking = _top_abs(qifa_rows, "delta_f1", count=5)
    qga_striking = _top_abs(qga_rows, "delta_f1", count=5)
    all_deltas = qifa_rows + qga_rows
    most_degraded = sorted(_rows_with_finite_delta(all_deltas, "delta_f1"), key=lambda row: float(row["delta_f1"]))[:5]
    most_improved = sorted(_rows_with_finite_delta(all_deltas, "delta_f1"), key=lambda row: float(row["delta_f1"]), reverse=True)[:5]

    lines = [
        "# Per-class IDS analysis annex",
        "",
        "## 1. Objective",
        "",
        "This annex adds class-level evidence for the E1-E8 reduced QI benchmark, complementing aggregate Macro-F1 with one-vs-rest metrics and confusion-matrix structure.",
        "",
        "## 2. Method",
        "",
        "Metrics were extracted from the existing best checkpoints for E1-E8 only; no training was relaunched. Each model was evaluated on the scenario validation set, using `global_val.npz` when present and otherwise the concatenation of `node1`, `node2`, and `node3` validation NPZ files. QGA-15 experiments applied the saved boolean feature mask before inference.",
        "",
        "## 3. Definitions",
        "",
        "For class c, TP_c = CM[c,c], FP_c = sum(CM[:,c]) - TP_c, FN_c = sum(CM[c,:]) - TP_c, and TN_c = total - TP_c - FP_c - FN_c. Undefined ratios are reported as NaN when their denominator is zero.",
        "",
        "## 4. Metrics used",
        "",
        "The annex reports per-class accuracy, precision, recall, F1 score, specificity, false-positive rate, false-negative rate, and support, plus raw and row-normalized 34-class confusion matrices.",
        "",
        "## 5. Weakest classes overall",
        "",
    ]
    for row in weakest:
        lines.append(f"- Class {row['class_id']} `{row['class_name']}`: mean F1={_format_float(row['mean_f1'])}")

    lines.extend(["", "## 6. Rare classes analysis", ""])
    for scenario, rows in rare.items():
        if rows:
            joined = ", ".join(f"{row['class_id']} `{row['class_name']}` (support={row['support']})" for row in rows)
        else:
            joined = "none below support < 1000"
        lines.append(f"- `{scenario}`: {joined}")

    lines.extend(["", "## 7. QIFA per-class effect", ""])
    for row in qifa_striking[:5]:
        direction = "improves" if float(row["delta_f1"]) > 0 else "degrades"
        lines.append(
            f"- {row['comparison']} ({row['scenario']}, {row['features']}), class {row['class_id']} `{row['class_name']}`: "
            f"QIFA {direction} F1 by {_format_float(row['delta_f1'])} and recall by {_format_float(row['delta_recall'])}."
        )

    lines.extend(["", "## 8. QGA-15 per-class effect", ""])
    for row in qga_striking[:5]:
        direction = "improves" if float(row["delta_f1"]) > 0 else "degrades"
        lines.append(
            f"- {row['comparison']} ({row['scenario']}, {row['features']}), class {row['class_id']} `{row['class_name']}`: "
            f"QGA-15 {direction} F1 by {_format_float(row['delta_f1'])} and recall by {_format_float(row['delta_recall'])}."
        )

    lines.extend(["", "## 9. Most degraded classes", ""])
    for row in most_degraded:
        lines.append(
            f"- {row['comparison']} class {row['class_id']} `{row['class_name']}`: delta F1={_format_float(row['delta_f1'])}"
        )

    lines.extend(["", "## 10. Most improved classes", ""])
    for row in most_improved:
        lines.append(
            f"- {row['comparison']} class {row['class_id']} `{row['class_name']}`: delta F1={_format_float(row['delta_f1'])}"
        )

    lines.extend(
        [
            "",
            "## 11. Limitations",
            "",
            "These are single-seed, validation-set-only metrics. They reuse the best checkpoint selected during the original benchmark, so the analysis is suitable as a reference annex but not as a new independent benchmark. QGA and QIFA effects can also interact with class support, so rare-class conclusions should be treated as diagnostic rather than definitive.",
            "",
        ]
    )
    (REPORT_DIR / "per_class_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    class_names = _load_class_names()
    scenario_cache: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    results: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []

    for spec in EXPERIMENTS:
        try:
            print(f"Evaluating {spec.experiment_id}: {spec.experiment_name}")
            results[spec.experiment_id] = evaluate_experiment(spec, class_names, scenario_cache)
        except Exception as exc:
            failures.append(
                {
                    "experiment_id": spec.experiment_id,
                    "experiment_name": spec.experiment_name,
                    "error": str(exc),
                }
            )

    if results:
        _, qifa_rows, qga_rows = _write_cross_experiment_tables(results)
        _write_figures(results, qifa_rows, qga_rows)
        _write_analysis_report(results, qifa_rows, qga_rows)

    status = {
        "completed": sorted(results.keys()),
        "failures": failures,
        "data_sources": {
            scenario: source for scenario, (_, _, source) in scenario_cache.items()
        },
        "checkpoints": {
            experiment_id: result["checkpoint_path"] for experiment_id, result in results.items()
        },
    }
    (REPORT_DIR / "evaluation_status.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"Per-class evaluation completed with failures: {failures}")
    print(f"Per-class metrics written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
