"""Matplotlib figures for P6 hierarchical Flower outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fl_hierarchical.summary_schema import architecture_string, figure_filenames


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_text(path: Path, title: str, message: str) -> None:
    plt = _plt()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _line(path: Path, rows: list[dict[str, Any]], keys: list[str], labels: list[str], title: str, ylabel: str) -> None:
    if not rows:
        _save_text(path, title, "No round metrics available.")
        return
    plt = _plt()
    path.parent.mkdir(parents=True, exist_ok=True)
    x = [int(row["round"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for key, label in zip(keys, labels):
        ax.plot(x, [float(row.get(key, 0.0)) for row in rows], marker="o", linewidth=1.7, label=label)
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _confusion(path: Path, matrix: list[list[int]], class_names: list[str], title: str) -> None:
    plt = _plt()
    arr = np.asarray(matrix, dtype=np.int64)
    size = 7.5 if len(class_names) <= 10 else 11.5
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(arr, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=90, fontsize=7 if len(class_names) > 10 else 9)
    ax.set_yticks(range(len(class_names)), labels=class_names, fontsize=7 if len(class_names) > 10 else 9)
    if len(class_names) <= 10:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, str(int(arr[i, j])), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _per_class_f1(path: Path, per_class: dict[str, Any], title: str) -> None:
    plt = _plt()
    rows = [per_class[key] for key in sorted(per_class, key=lambda item: int(item))]
    names = [row["class_name"] for row in rows]
    values = [float(row["f1"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.32), 5))
    ax.bar(range(len(names)), values, color="#2b8cbe")
    ax.set_title(title)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(names)), labels=names, rotation=90, fontsize=7 if len(names) > 10 else 9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _one_vs_rest(path: Path, per_class: dict[str, Any], title: str) -> None:
    plt = _plt()
    rows = [per_class[key] for key in sorted(per_class, key=lambda item: int(item))]
    names = [row["class_name"] for row in rows]
    metrics = ["TP", "FP", "TN", "FN"]
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 0.42), 5))
    for offset, metric in enumerate(metrics):
        ax.bar(x + (offset - 1.5) * width, [int(row[metric]) for row in rows], width, label=metric)
    ax.set_title(title)
    ax.set_ylabel("Samples")
    ax.set_xticks(x, labels=names, rotation=90, fontsize=7 if len(names) > 10 else 9)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _client_heatmap(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    if not rows:
        _save_text(path, title, "No client metrics available.")
        return
    clients = sorted({str(row["client_id"]) for row in rows})
    metrics = ["local_accuracy", "local_macro_f1", "local_recall_macro", "local_fpr_macro"]
    values = np.zeros((len(clients), len(metrics)), dtype=float)
    for i, client_id in enumerate(clients):
        client_rows = [row for row in rows if str(row["client_id"]) == client_id]
        for j, metric in enumerate(metrics):
            values[i, j] = float(np.mean([float(row.get(metric, 0.0)) for row in client_rows]))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=max(1.0, float(values.max())))
    ax.set_xticks(range(len(metrics)), labels=["accuracy", "macro F1", "macro recall", "macro FPR"])
    ax.set_yticks(range(len(clients)), labels=clients)
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _top_errors(path: Path, pairs: list[dict[str, Any]]) -> None:
    if not pairs:
        _save_text(path, "L3 attack-type top errors", "No misclassification pairs available.")
        return
    plt = _plt()
    labels = [f"{row['true_class']} -> {row['pred_class']}" for row in pairs]
    values = [int(row["count"]) for row in pairs]
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(labels) * 0.32)))
    ax.barh(range(len(labels)), values, color="#756bb1")
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("L3 attack-type top confusion pairs")
    ax.set_xlabel("Count")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _architecture(path: Path, task: str, model_cfg: dict[str, Any], num_parameters: int) -> None:
    title = "P6 L2/L3 Hierarchical MLP"
    text = (
        f"{task}\n\n"
        f"Architecture: {architecture_string(model_cfg)}\n"
        f"Activation: {model_cfg.get('activation', 'relu')}\n"
        f"Dropout: {model_cfg.get('dropout', 0.0)}\n"
        f"Parameters: {num_parameters}\n\n"
        "Flower FedAvg runtime; L2/L3 are experimental and not deployed."
    )
    _save_text(path, title, text)


def _tree(path: Path) -> None:
    _save_text(
        path,
        "Hierarchical IDS L1/L2/L3",
        "L1 production: normal vs attack\nL2 experimental: attack family\nL3 experimental: attack type",
    )


def _pipeline(path: Path) -> None:
    _save_text(
        path,
        "Hierarchical inference pipeline",
        "Input features -> L1 binary gate -> optional L2 family analysis -> optional L3 attack-type analysis\nOnly L1 is planned for dashboard production.",
    )


def _comparison(path: Path, comparison: dict[str, Any]) -> None:
    plt = _plt()
    labels = ["L1/P4 macro F1", "P6 macro F1", "L1/P4 accuracy", "P6 accuracy"]
    values = [
        float(comparison.get("p4_l1_macro_f1", 0.0)),
        float(comparison.get("p6_macro_f1", 0.0)),
        float(comparison.get("p4_l1_accuracy", 0.0)),
        float(comparison.get("p6_accuracy", 0.0)),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(labels, values, color=["#31a354", "#2b8cbe", "#31a354", "#2b8cbe"])
    ax.set_ylim(0, max(1.0, max(values) * 1.1 if values else 1.0))
    ax.set_title("L1/P4 reference vs P6 experimental task")
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _summary_table(path: Path, metrics_test: dict[str, Any], task: str) -> None:
    text = (
        f"Task: {task}\n"
        f"Accuracy: {float(metrics_test.get('accuracy', 0.0)):.4f}\n"
        f"Macro-F1: {float(metrics_test.get('macro_f1', 0.0)):.4f}\n"
        f"Weighted-F1: {float(metrics_test.get('weighted_f1', 0.0)):.4f}\n"
        f"Macro recall: {float(metrics_test.get('recall_macro', 0.0)):.4f}"
    )
    _save_text(path, "P6 L2/L3 summary table", text)


def generate_hierarchical_figures(
    *,
    figures_dir: Path,
    task: str,
    round_rows: list[dict[str, Any]],
    client_rows: list[dict[str, Any]],
    metrics_test: dict[str, Any],
    comparison: dict[str, Any],
    model_cfg: dict[str, Any],
    num_parameters: int,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: figures_dir / name for name in figure_filenames(task)}
    prefix = "l2_family" if task == "l2_family" else "l3_attack_type"
    if task == "l2_family":
        _confusion(paths["l2_family_confusion_matrix.png"], metrics_test["confusion_matrix"], list(metrics_test["class_names"]), "L2 family confusion matrix")
        _per_class_f1(paths["l2_family_per_class_f1.png"], metrics_test["per_class"], "L2 family per-class F1")
        _one_vs_rest(paths["l2_family_one_vs_rest_tp_fp_tn_fn.png"], metrics_test["per_class"], "L2 one-vs-rest TP/FP/TN/FN")
        _line(paths["l2_family_metrics_by_round.png"], round_rows, ["macro_f1", "accuracy"], ["macro F1", "accuracy"], "L2 metrics by round", "Metric")
        _line(paths["l2_family_bandwidth_by_round.png"], round_rows, ["communication_total_bytes", "communication_cumulative_bytes"], ["round bytes", "cumulative bytes"], "L2 bandwidth by round", "Bytes")
        _client_heatmap(paths["l2_family_client_metrics_heatmap.png"], client_rows, "L2 client metrics heatmap")
    else:
        _top_errors(paths["l3_attack_type_top_errors.png"], metrics_test.get("top_confusion_pairs", []))
        _per_class_f1(paths["l3_attack_type_per_class_f1.png"], metrics_test["per_class"], "L3 attack-type per-class F1")
        _one_vs_rest(paths["l3_attack_type_one_vs_rest_tp_fp_tn_fn.png"], metrics_test["per_class"], "L3 one-vs-rest TP/FP/TN/FN")
        _line(paths["l3_attack_type_metrics_by_round.png"], round_rows, ["macro_f1", "accuracy"], ["macro F1", "accuracy"], "L3 metrics by round", "Metric")
        _line(paths["l3_attack_type_bandwidth_by_round.png"], round_rows, ["communication_total_bytes", "communication_cumulative_bytes"], ["round bytes", "cumulative bytes"], "L3 bandwidth by round", "Bytes")
        _confusion(paths["l3_attack_type_confusion_matrix.png"], metrics_test["confusion_matrix"], list(metrics_test["class_names"]), "L3 attack-type confusion matrix")
    _tree(paths["hierarchical_tree_l1_l2_l3.png"])
    _pipeline(paths["hierarchical_inference_pipeline.png"])
    _comparison(paths["l1_l2_l3_comparison.png"], comparison)
    _summary_table(paths["l2_l3_summary_table.png"], metrics_test, prefix)
    _architecture(paths[f"{prefix}_architecture.png"], task, model_cfg, num_parameters)
    return [paths[name] for name in figure_filenames(task) if paths[name].exists()]
