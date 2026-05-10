"""Figure generation for P9 QIFA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def generate_qifa_figures(
    *,
    output_dir: Path,
    round_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    comparison: dict[str, Any],
    confusion_metrics: dict[str, Any],
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[str] = []
    rounds = [int(row["round"]) for row in round_rows] or [1]
    macro_f1 = [float(row.get("macro_f1", 0.0)) for row in round_rows] or [0.0]
    fpr = [float(row.get("FPR", 0.0)) for row in round_rows] or [0.0]
    loss = [float(row.get("val_loss_mean", 0.0)) for row in round_rows] or [0.0]
    entropies = [float(row.get("qifa_entropy", 0.0)) for row in round_rows] or [0.0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, [float(row.get("mean_final_weight", 0.0)) for row in round_rows] or [0.0], marker="o")
    ax.set_title("QIFA Weights By Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean final weight")
    figure_paths.append(_save(fig, output_dir / "qifa_weights_by_round.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    if score_rows:
        clients = sorted({str(row["client_id"]) for row in score_rows})
        client_means = [np.mean([float(row["score"]) for row in score_rows if row["client_id"] == client_id]) for client_id in clients]
        ax.bar(clients, client_means)
    ax.set_title("QIFA Scores By Client")
    ax.set_ylabel("Mean score")
    figure_paths.append(_save(fig, output_dir / "qifa_scores_by_client.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, [float(row.get("max_probability", 0.0)) for row in round_rows] or [0.0], marker="o")
    ax.set_title("QIFA Probabilities By Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Max client probability")
    figure_paths.append(_save(fig, output_dir / "qifa_probabilities_by_round.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    p5_macro = comparison.get("p5_macro_f1")
    ax.plot(rounds, macro_f1, marker="o", label="QIFA")
    if p5_macro not in (None, ""):
        ax.axhline(float(p5_macro), color="tab:orange", linestyle="--", label="P5 FedAvg")
    ax.set_title("QIFA vs FedAvg Macro-F1")
    ax.legend()
    figure_paths.append(_save(fig, output_dir / "qifa_vs_fedavg_macro_f1.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    p5_fpr = comparison.get("p5_fpr")
    ax.plot(rounds, fpr, marker="o", label="QIFA")
    if p5_fpr not in (None, ""):
        ax.axhline(float(p5_fpr), color="tab:orange", linestyle="--", label="P5 FedAvg")
    ax.set_title("QIFA vs FedAvg FPR")
    ax.legend()
    figure_paths.append(_save(fig, output_dir / "qifa_vs_fedavg_fpr.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, loss, marker="o")
    ax.set_title("QIFA Convergence Loss")
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation loss")
    figure_paths.append(_save(fig, output_dir / "qifa_convergence_loss.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    if score_rows:
        clients = sorted({str(row["client_id"]) for row in score_rows})
        rounds_sorted = sorted({int(row["round"]) for row in score_rows})
        heat = np.zeros((len(clients), len(rounds_sorted)), dtype=float)
        for i, client_id in enumerate(clients):
            for j, round_number in enumerate(rounds_sorted):
                values = [float(row["final_weight"]) for row in score_rows if row["client_id"] == client_id and int(row["round"]) == round_number]
                heat[i, j] = values[0] if values else 0.0
        im = ax.imshow(heat, aspect="auto")
        ax.set_yticks(range(len(clients)), labels=clients)
        ax.set_xticks(range(len(rounds_sorted)), labels=rounds_sorted)
        fig.colorbar(im, ax=ax)
    ax.set_title("QIFA Client Contribution Heatmap")
    figure_paths.append(_save(fig, output_dir / "qifa_client_contribution_heatmap.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["alpha"], [float(np.mean(macro_f1))])
    ax.set_title("QIFA Alpha Robustness Heatmap")
    figure_paths.append(_save(fig, output_dir / "qifa_alpha_robustness_heatmap.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, entropies, marker="o")
    ax.set_title("QIFA Gamma Sensitivity")
    ax.set_xlabel("Round")
    ax.set_ylabel("Entropy")
    figure_paths.append(_save(fig, output_dir / "qifa_gamma_sensitivity.png"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.table(
        cellText=[
            ["Macro-F1", f"{comparison.get('p9_macro_f1', 0.0):.4f}" if comparison.get("p9_macro_f1") is not None else "n/a"],
            ["Gap vs P5", f"{comparison.get('gap_macro_f1_vs_p5', 0.0):.4f}" if comparison.get("gap_macro_f1_vs_p5") is not None else "n/a"],
            ["FPR", f"{confusion_metrics.get('FPR', 0.0):.4f}" if confusion_metrics.get("FPR") is not None else "n/a"],
        ],
        colLabels=["Metric", "Value"],
        loc="center",
    )
    ax.set_title("P9 Ablation Summary")
    figure_paths.append(_save(fig, output_dir / "p9_ablation_summary.png"))
    return figure_paths
