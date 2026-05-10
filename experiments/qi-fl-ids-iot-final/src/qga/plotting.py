"""Matplotlib figures for P8 QGA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: str | Path) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out.as_posix()


def plot_qga_figures(
    *,
    history: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
    mask: np.ndarray,
    feature_names: list[str],
    figures_dir: str | Path,
) -> list[str]:
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    generations = sorted({int(row["generation"]) for row in history})
    best_by_generation = [
        max(float(row["fitness"]) for row in history if int(row["generation"]) == generation)
        for generation in generations
    ]
    avg_features = [
        float(np.mean([int(row["features_count"]) for row in history if int(row["generation"]) == generation]))
        for generation in generations
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(generations, best_by_generation, marker="o")
    ax.set_title("QGA fitness evolution")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    ax.grid(alpha=0.3)
    paths.append(_save(fig, out / "qga_fitness_evolution.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(generations, avg_features, marker="o", color="#3b7ddd")
    ax.set_title("QGA selected feature count")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Average selected features")
    ax.grid(alpha=0.3)
    paths.append(_save(fig, out / "qga_num_features_evolution.png"))

    selected = [feature_names[idx] for idx in np.flatnonzero(mask == 1)]
    fig, ax = plt.subplots(figsize=(8, max(4, len(selected) * 0.25)))
    ax.barh(selected, [1] * len(selected), color="#1f77b4")
    ax.set_title("Selected features")
    ax.set_xlabel("Selected")
    paths.append(_save(fig, out / "qga_selected_features_barplot.png"))

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.imshow(mask.reshape(1, -1), aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
    ax.set_title("QGA feature mask")
    paths.append(_save(fig, out / "qga_feature_mask.png"))

    top = ranking[: min(20, len(ranking))]
    labels = [feature_names[int(row["feature_index"])] for row in top]
    values = [float(row["selection_frequency"]) for row in top]
    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.25)))
    ax.barh(labels[::-1], values[::-1], color="#2ca02c")
    ax.set_title("QGA feature ranking")
    ax.set_xlabel("Selection frequency")
    paths.append(_save(fig, out / "qga_feature_importance_ranking.png"))
    return paths


def plot_binary_adapter_figures(
    *,
    metrics_rounds: list[dict[str, Any]],
    confusion_metrics: dict[str, Any],
    output_dir: str | Path,
    prefix: str,
) -> list[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    rounds = [int(row["round"]) for row in metrics_rounds]
    macro = [float(row.get("macro_f1", 0.0)) for row in metrics_rounds]
    bandwidth = [float(row.get("communication_cumulative_bytes", row.get("bandwidth_total_bytes", 0.0))) for row in metrics_rounds]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, macro, marker="o")
    ax.set_title(f"{prefix} Macro-F1 by round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Macro-F1")
    ax.grid(alpha=0.3)
    paths.append(_save(fig, out / f"{prefix}_macro_f1_by_round.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, bandwidth, marker="o", color="#9467bd")
    ax.set_title(f"{prefix} bandwidth by round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative bytes")
    ax.grid(alpha=0.3)
    paths.append(_save(fig, out / f"{prefix}_bandwidth_by_round.png"))

    matrix = np.array(
        [
            [int(confusion_metrics.get("TN", 0)), int(confusion_metrics.get("FP", 0))],
            [int(confusion_metrics.get("FN", 0)), int(confusion_metrics.get("TP", 0))],
        ]
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["normal", "attack"])
    ax.set_yticks([0, 1], labels=["normal", "attack"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    ax.set_title(f"{prefix} confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    paths.append(_save(fig, out / f"{prefix}_confusion_matrix.png"))
    return paths
