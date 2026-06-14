"""Matplotlib figures for P7 HeteroFL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from multitier_heterofl.summary_schema import FIGURES


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _text(path: Path, title: str, message: str) -> None:
    plt = _plt()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _line(path: Path, rows: list[dict[str, Any]], key: str, title: str, ylabel: str) -> None:
    if not rows:
        _text(path, title, "No round metrics available.")
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = [int(row["round"]) for row in rows]
    ax.plot(x, [float(row.get(key, 0.0)) for row in rows], marker="o")
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(labels, values, color="#2b8cbe")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _confusion(path: Path, matrix: list[list[int]], labels: list[str], title: str) -> None:
    plt = _plt()
    arr = np.asarray(matrix, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(7 if len(labels) < 10 else 10, 6 if len(labels) < 10 else 10))
    image = ax.imshow(arr, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def generate_figures(
    *,
    figures_dir: Path,
    task: str,
    round_rows: list[dict[str, Any]],
    tier_rows: list[dict[str, Any]],
    metrics_test: dict[str, Any],
    comparison: dict[str, Any],
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: figures_dir / name for name in FIGURES}
    _text(paths["multitier_architecture.png"], "P7 Multi-tier HeteroFL", "weak: 28->64->out\nmedium: 28->128->64->out\npowerful/supernet: 28->256->128->out")
    _text(paths["supernet_slicing.png"], "Shared-supernet prefix slicing", "Clients train prefix neuron slices. Uncovered slices keep previous global values.")
    labels = [row["tier"] for row in tier_rows]
    _bar(paths["tier_parameter_comparison.png"], labels, [float(row["num_parameters"]) for row in tier_rows], "Parameters by tier", "Parameters")
    _bar(paths["tier_model_size_comparison.png"], labels, [float(row["model_size_bytes"]) for row in tier_rows], "Model size by tier", "Bytes")
    _bar(paths["tier_bandwidth_comparison.png"], labels, [float(row["bandwidth_total_bytes"]) for row in tier_rows], "Bandwidth by tier", "Bytes")
    _bar(paths["tier_latency_comparison.png"], labels, [float(row.get("avg_latency_ms_per_sample", 0.0)) for row in tier_rows], "Latency by tier", "ms/sample")
    if task == "l1_binary":
        _bar(paths["l1_p4_p5_p7_comparison.png"], ["P4", "P5", "P7"], [float(comparison.get("p4_macro_f1", 0.0)), float(comparison.get("p5_macro_f1", 0.0)), float(metrics_test.get("macro_f1", 0.0))], "L1 P4/P5/P7 macro-F1", "Macro-F1")
        _line(paths["l1_macro_f1_p5_vs_p7_by_round.png"], round_rows, "macro_f1", "L1 P7 macro-F1 by round", "Macro-F1")
        _line(paths["l1_fpr_p5_vs_p7_by_round.png"], round_rows, "FPR", "L1 P7 FPR by round", "FPR")
        _confusion(paths["l1_multitier_confusion_matrix.png"], metrics_test["confusion_matrix"], ["normal", "attack"], "L1 HeteroFL confusion matrix")
    else:
        _bar(paths["l2_p6_vs_p7_comparison.png"], ["P6", "P7"], [float(comparison.get("p6_macro_f1", 0.0)), float(metrics_test.get("macro_f1", 0.0))], "L2 P6/P7 macro-F1", "Macro-F1")
        _line(paths["l2_macro_f1_by_round.png"], round_rows, "macro_f1", "L2 P7 macro-F1 by round", "Macro-F1")
        _confusion(paths["l2_family_confusion_matrix.png"], metrics_test["confusion_matrix"], list(metrics_test.get("class_names", [])), "L2 family confusion matrix")
        per_class = metrics_test.get("per_class", {})
        names = [per_class[k]["class_name"] for k in sorted(per_class, key=lambda item: int(item))]
        values = [float(per_class[k]["f1"]) for k in sorted(per_class, key=lambda item: int(item))]
        _bar(paths["l2_per_family_f1.png"], names, values, "L2 per-family F1", "F1")
    for name in [
        "heatmap_p7_l1_macro_f1_alpha_k.png",
        "heatmap_p7_l1_fpr_alpha_k.png",
        "heatmap_p7_l2_macro_f1_alpha_k.png",
        "barplot_p7_bandwidth_by_scenario.png",
        "p7_scenario_ranking_table.png",
    ]:
        _text(paths[name], name.replace("_", " "), "Generated after sequential grid aggregation.")
    return [path for path in paths.values() if path.exists()]
