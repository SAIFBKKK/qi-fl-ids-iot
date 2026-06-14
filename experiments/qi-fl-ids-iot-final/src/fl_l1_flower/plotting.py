"""Matplotlib figures for P5.2 Flower runtime reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fl_l1_flower.summary_schema import FIGURE_FILENAMES, architecture_string


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_placeholder(path: Path, title: str, message: str) -> None:
    plt = _plt()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _line_plot(path: Path, rows: list[dict[str, Any]], keys: list[str], labels: list[str], title: str, ylabel: str) -> None:
    if not rows:
        _save_placeholder(path, title, "No round metrics available.")
        return
    plt = _plt()
    path.parent.mkdir(parents=True, exist_ok=True)
    x = [int(row["round"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label in zip(keys, labels):
        ax.plot(x, [float(row.get(key, 0.0)) for row in rows], marker="o", linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _confusion_matrix(path: Path, metrics: dict[str, Any]) -> None:
    plt = _plt()
    matrix = np.array(
        [
            [int(metrics.get("TN", 0)), int(metrics.get("FP", 0))],
            [int(metrics.get("FN", 0)), int(metrics.get("TP", 0))],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred normal", "pred attack"])
    ax.set_yticks([0, 1], labels=["true normal", "true attack"])
    ax.set_title("Flower L1 confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _threshold_sweep(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        _save_placeholder(path, "Flower L1 threshold sweep", "No threshold sweep available.")
        return
    plt = _plt()
    thresholds = [float(row["threshold"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label in [("f1_attack", "attack F1"), ("recall_attack", "attack recall"), ("FPR", "FPR")]:
        ax.plot(thresholds, [float(row.get(key, 0.0)) for row in rows], marker="o", linewidth=1.5, label=label)
    ax.set_title("Flower L1 validation threshold sweep")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _tp_tn_fp_fn(path: Path, metrics: dict[str, Any]) -> None:
    plt = _plt()
    labels = ["TP", "TN", "FP", "FN"]
    values = [int(metrics.get(label, 0)) for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color=["#2b8cbe", "#31a354", "#de2d26", "#756bb1"])
    ax.set_title("Flower L1 TP/TN/FP/FN")
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _roc_pr(path_roc: Path, path_pr: Path, y_true: np.ndarray, prob_attack: np.ndarray) -> None:
    try:
        from sklearn.metrics import precision_recall_curve, roc_curve
    except Exception as exc:  # pragma: no cover - optional dependency
        _save_placeholder(path_roc, "Flower L1 ROC curve", f"sklearn unavailable: {exc}")
        _save_placeholder(path_pr, "Flower L1 PR curve", f"sklearn unavailable: {exc}")
        return
    plt = _plt()
    fpr, tpr, _ = roc_curve(y_true, prob_attack)
    precision, recall, _ = precision_recall_curve(y_true, prob_attack)

    for path, x, y, title, xlabel, ylabel in [
        (path_roc, fpr, tpr, "Flower L1 ROC curve", "FPR", "TPR"),
        (path_pr, recall, precision, "Flower L1 PR curve", "Recall", "Precision"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.plot(x, y, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)


def _comparison(path: Path, comparison: dict[str, Any]) -> None:
    plt = _plt()
    labels = ["accuracy", "macro F1", "attack recall", "FPR"]
    p4 = [
        float(comparison.get("p4_accuracy", 0.0)),
        float(comparison.get("p4_macro_f1", 0.0)),
        float(comparison.get("p4_attack_recall", 0.0)),
        float(comparison.get("p4_fpr", 0.0)),
    ]
    p5 = [
        float(comparison.get("p5_2_accuracy", 0.0)),
        float(comparison.get("p5_2_macro_f1", 0.0)),
        float(comparison.get("p5_2_attack_recall", 0.0)),
        float(comparison.get("p5_2_fpr", 0.0)),
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, p4, width, label="P4 centralized")
    ax.bar(x + width / 2, p5, width, label="P5.2 Flower")
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, max(1.0, max(p4 + p5) * 1.1))
    ax.set_title("Flower L1 vs P4 centralized")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _client_heatmap(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        _save_placeholder(path, "Flower client metrics heatmap", "No client metrics available.")
        return
    clients = sorted({str(row["client_id"]) for row in rows})
    metrics = ["local_macro_f1", "local_attack_recall", "local_fpr"]
    values = np.zeros((len(clients), len(metrics)), dtype=float)
    for i, client_id in enumerate(clients):
        client_rows = [row for row in rows if str(row["client_id"]) == client_id]
        for j, metric in enumerate(metrics):
            values[i, j] = float(np.mean([float(row.get(metric, 0.0)) for row in client_rows]))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(metrics)), labels=["macro F1", "attack recall", "FPR"])
    ax.set_yticks(range(len(clients)), labels=clients)
    ax.set_title("Flower client metrics heatmap")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _architecture(path: Path, model_cfg: dict[str, Any], num_parameters: int) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")
    text = (
        "FlowerL1FedAvgMLP\n\n"
        f"Architecture: {architecture_string(model_cfg)}\n"
        f"Activation: {model_cfg.get('activation', 'relu')}\n"
        f"Dropout: {model_cfg.get('dropout', 0.0)}\n"
        f"Parameters: {num_parameters}"
    )
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=13, bbox={"boxstyle": "round", "facecolor": "#f2f2f2"})
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def generate_flower_figures(
    *,
    figures_dir: Path,
    round_rows: list[dict[str, Any]],
    client_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    metrics_test: dict[str, Any],
    comparison: dict[str, Any],
    y_true_test: np.ndarray,
    prob_attack_test: np.ndarray,
    model_cfg: dict[str, Any],
    num_parameters: int,
) -> list[Path]:
    """Generate the complete expected Flower figure set and return paths."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: figures_dir / name for name in FIGURE_FILENAMES}
    _line_plot(paths["fl_l1_flower_loss_by_round.png"], round_rows, ["train_loss_mean", "val_loss_mean"], ["train loss", "val loss"], "Flower L1 loss by round", "Loss")
    _line_plot(paths["fl_l1_flower_macro_f1_by_round.png"], round_rows, ["macro_f1"], ["macro F1"], "Flower L1 macro F1 by round", "Macro F1")
    _line_plot(paths["fl_l1_flower_attack_recall_by_round.png"], round_rows, ["attack_recall"], ["attack recall"], "Flower L1 attack recall by round", "Recall")
    _line_plot(paths["fl_l1_flower_fpr_by_round.png"], round_rows, ["FPR"], ["FPR"], "Flower L1 FPR by round", "FPR")
    _line_plot(paths["fl_l1_flower_bandwidth_by_round.png"], round_rows, ["communication_total_bytes", "communication_cumulative_bytes"], ["round bytes", "cumulative bytes"], "Flower L1 bandwidth by round", "Bytes")
    _line_plot(paths["fl_l1_flower_round_time_by_round.png"], round_rows, ["round_time_sec", "aggregation_time_sec"], ["round time", "aggregation time"], "Flower L1 runtime by round", "Seconds")
    _confusion_matrix(paths["fl_l1_flower_confusion_matrix.png"], metrics_test)
    _threshold_sweep(paths["fl_l1_flower_threshold_sweep.png"], threshold_rows)
    _tp_tn_fp_fn(paths["fl_l1_flower_tp_tn_fp_fn.png"], metrics_test)
    _roc_pr(paths["fl_l1_flower_roc_curve.png"], paths["fl_l1_flower_pr_curve.png"], y_true_test, prob_attack_test)
    _comparison(paths["fl_l1_flower_vs_p4.png"], comparison)
    _client_heatmap(paths["fl_l1_flower_client_metrics_heatmap.png"], client_rows)
    _architecture(paths["fl_l1_flower_architecture.png"], model_cfg, num_parameters)
    return [paths[name] for name in FIGURE_FILENAMES]
