"""P10 robustness plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def plot_summary(rows: list[dict[str, Any]], figures_dir: Path, clean_vs_poisoned: list[dict[str, Any]] | None = None) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    if not rows:
        return generated
    metrics = [
        ("macro_f1", "p10_macro_f1_vs_poison_rate.png", "Macro-F1 vs poison rate"),
        ("attack_recall", "p10_attack_recall_vs_poison_rate.png", "Attack recall vs poison rate"),
        ("fpr", "p10_fpr_vs_poison_rate.png", "FPR vs poison rate"),
        ("robustness_score", "p10_robustness_score_by_method.png", "Robustness score by method"),
    ]
    for key, filename, title in metrics:
        path = figures_dir / filename
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in sorted({row["method"] for row in rows}):
            selected = [row for row in rows if row["method"] == method and row.get(key) not in ("", None)]
            if not selected:
                continue
            selected.sort(key=lambda row: _to_float(row["poison_rate"]))
            ax.plot([_to_float(row["poison_rate"]) for row in selected], [_to_float(row[key]) for row in selected], marker="o", label=method)
        ax.set_xlabel("Poison rate")
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(path)

    if clean_vs_poisoned:
        path = figures_dir / "p10_clean_vs_poisoned_comparison.png"
        methods = [row["method"] for row in clean_vs_poisoned]
        clean = [_to_float(row["clean_macro_f1"]) for row in clean_vs_poisoned]
        poisoned = [_to_float(row["poisoned_macro_f1"]) for row in clean_vs_poisoned]
        x = range(len(methods))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar([i - 0.18 for i in x], clean, width=0.36, label="clean", color="#64748B")
        ax.bar([i + 0.18 for i in x], poisoned, width=0.36, label="poisoned", color="#DC2626")
        ax.set_xticks(list(x), methods, rotation=18, ha="right")
        ax.set_ylabel("Macro-F1")
        ax.set_title("Clean vs poisoned Macro-F1")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(path)

        path = figures_dir / "p10_best_method_under_poisoning.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        sorted_rows = sorted(rows, key=lambda row: _to_float(row.get("robustness_score")), reverse=True)
        ax.bar([row["method"] for row in sorted_rows], [_to_float(row["robustness_score"]) for row in sorted_rows], color="#16A34A")
        ax.set_ylabel("Robustness score")
        ax.set_title("Best method under label-flip poisoning")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(path)

    for filename, title in [
        ("p10_qifa_weights_under_attack.png", "QIFA weights under attack"),
    ]:
        path = figures_dir / filename
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.02, 0.55, title, fontsize=14, weight="bold")
        ax.text(0.02, 0.35, "No QIFA weight time-series artifact is available in the P10 in-process full summaries.", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(path)
    return generated
