"""P11 FedTN/MPS plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def plot_summary(rows: list[dict[str, Any]], figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    if not rows:
        return generated
    rows = sorted(rows, key=lambda row: (str(row.get("base_model")), int(row.get("rank", 0))))

    specs = [
        ("compressed_model_size_bytes", "p11_model_size_vs_rank.png", "Model size vs rank", "bytes"),
        ("bandwidth_reduction_ratio", "p11_bandwidth_reduction_by_rank.png", "Bandwidth reduction by rank", "ratio"),
        ("compression_ratio", "p11_best_rank_tradeoff.png", "Compression ratio by rank", "ratio"),
    ]
    for key, filename, title, ylabel in specs:
        path = figures_dir / filename
        fig, ax = plt.subplots(figsize=(8, 5))
        for base_model in sorted({row["base_model"] for row in rows}):
            selected = [row for row in rows if row["base_model"] == base_model]
            ax.plot([int(row["rank"]) for row in selected], [_f(row[key]) for row in selected], marker="o", label=base_model)
        ax.set_xlabel("Rank")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(path)

    for filename, title in [
        ("p11_macro_f1_vs_compression_ratio.png", "Macro-F1 vs compression ratio"),
        ("p11_attack_recall_vs_compression_ratio.png", "Attack recall vs compression ratio"),
    ]:
        path = figures_dir / filename
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.02, 0.55, title, fontsize=14, weight="bold")
        ax.text(0.02, 0.35, "Checkpoint evaluation not available in dry-run mode.", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(path)
    return generated
