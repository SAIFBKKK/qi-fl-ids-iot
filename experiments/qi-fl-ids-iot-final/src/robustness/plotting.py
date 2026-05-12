"""P10 robustness plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_summary(rows: list[dict[str, Any]], figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    if not rows:
        return generated
    metrics = [
        ("macro_f1", "p10_macro_f1_vs_poison_rate.png", "Macro-F1 vs poison rate"),
        ("attack_recall", "p10_attack_recall_vs_poison_rate.png", "Attack recall vs poison rate"),
        ("FPR", "p10_fpr_vs_poison_rate.png", "FPR vs poison rate"),
        ("robustness_score", "p10_robustness_score_by_method.png", "Robustness score by method"),
    ]
    for key, filename, title in metrics:
        path = figures_dir / filename
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in sorted({row["method"] for row in rows}):
            selected = [row for row in rows if row["method"] == method and row.get(key) not in ("", None)]
            if not selected:
                continue
            selected.sort(key=lambda row: float(row["poison_rate"]))
            ax.plot([float(row["poison_rate"]) for row in selected], [float(row[key]) for row in selected], marker="o", label=method)
        ax.set_xlabel("Poison rate")
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(path)

    for filename, title in [
        ("p10_clean_vs_poisoned_comparison.png", "Clean vs poisoned comparison"),
        ("p10_qifa_weights_under_attack.png", "QIFA weights under attack"),
    ]:
        path = figures_dir / filename
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.02, 0.55, title, fontsize=14, weight="bold")
        ax.text(0.02, 0.35, "Generated as a placeholder until full robustness runs are available.", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(path)
    return generated
