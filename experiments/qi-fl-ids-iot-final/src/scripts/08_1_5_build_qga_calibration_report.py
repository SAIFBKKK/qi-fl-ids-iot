"""Build P8.1.5 QGA calibration report and figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qga.calibration import rank_masks_from_short_validation
from qga.config import load_config, load_json, repo_path, write_json


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.as_posix()


def _plot_sweep(sweep: list[dict[str, Any]], figures_dir: Path) -> list[str]:
    paths: list[str] = []
    if not sweep:
        return paths
    profiles = sorted({row["profile"] for row in sweep})
    macro_by_profile = [
        np.mean([_float(row, "validation_macro_f1") for row in sweep if row["profile"] == profile])
        for profile in profiles
    ]
    features_by_profile = [
        np.mean([_float(row, "features_count") for row in sweep if row["profile"] == profile])
        for profile in profiles
    ]
    fpr_by_profile = [
        np.mean([_float(row, "validation_fpr") for row in sweep if row["profile"] == profile])
        for profile in profiles
    ]
    fitness_groups = [[_float(row, "fitness") for row in sweep if row["profile"] == profile] for profile in profiles]

    for name, values, ylabel, title in [
        ("qga_profiles_macro_f1.png", macro_by_profile, "Validation Macro-F1", "QGA profiles Macro-F1"),
        ("qga_profiles_features_count.png", features_by_profile, "Features", "QGA profiles feature count"),
        ("qga_profiles_fpr.png", fpr_by_profile, "Validation FPR", "QGA profiles FPR"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(profiles, values, color="#2f6fbb")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        paths.append(_save(fig, figures_dir / name))

    fig, ax = plt.subplots(figsize=(8, 4))
    try:
        ax.boxplot(fitness_groups, tick_labels=profiles)
    except TypeError:  # pragma: no cover - older matplotlib
        ax.boxplot(fitness_groups, labels=profiles)
    ax.set_title("QGA profile fitness distribution")
    ax.set_ylabel("Fitness")
    ax.tick_params(axis="x", rotation=20)
    paths.append(_save(fig, figures_dir / "qga_profile_fitness_boxplot.png"))
    return paths


def _plot_short(short: list[dict[str, Any]], ranking: list[dict[str, Any]], figures_dir: Path) -> list[str]:
    paths: list[str] = []
    if not short:
        return paths
    labels = [f"{row['mask_id']}\n{row['scenario']}" for row in short]
    for name, key, ylabel in [
        ("qga_short_flower_macro_f1_by_scenario.png", "val_macro_f1", "Validation Macro-F1"),
        ("qga_short_flower_fpr_by_scenario.png", "val_fpr", "Validation FPR"),
    ]:
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 4))
        ax.bar(labels, [_float(row, key) for row in short], color="#4477aa")
        ax.set_ylabel(ylabel)
        ax.set_title(name.replace(".png", "").replace("_", " "))
        ax.tick_params(axis="x", rotation=45)
        paths.append(_save(fig, figures_dir / name))
    if ranking:
        fig, ax = plt.subplots(figsize=(9, max(4, len(ranking) * 0.35)))
        names = [row["mask_id"] for row in ranking]
        scores = [float(row["engineering_score"]) for row in ranking]
        ax.barh(names[::-1], scores[::-1], color="#33a02c")
        ax.set_xlabel("Engineering score")
        ax.set_title("QGA engineering score ranking")
        paths.append(_save(fig, figures_dir / "qga_engineering_score_ranking.png"))
    return paths


def _plot_stability(config: dict[str, Any], figures_dir: Path) -> list[str]:
    stability_path = repo_path(config, "outputs.reports_dir") / "p8_qga_mask_stability.json"
    if not stability_path.exists():
        return []
    stability = load_json(stability_path)
    labels = [row["feature"] for row in stability]
    values = np.asarray([[float(row["selection_frequency"]) for row in stability]], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(values, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_title("QGA mask stability heatmap")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    return [_save(fig, figures_dir / "qga_mask_stability_heatmap.png")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    reports_dir = repo_path(config, "outputs.reports_dir")
    figures_dir = repo_path(config, "outputs.figures_dir") / "calibration"
    sweep = _read_csv(reports_dir / "p8_qga_profile_sweep_summary.csv")
    short = _read_csv(reports_dir / "p8_qga_flower_short_validation.csv")
    score_weights = {key: float(value) for key, value in config["qga_calibration"]["engineering_score"].items()}
    ranking = rank_masks_from_short_validation(short, score_weights=score_weights) if short else []
    figures = []
    figures.extend(_plot_sweep(sweep, figures_dir))
    figures.extend(_plot_short(short, ranking, figures_dir))
    figures.extend(_plot_stability(config, figures_dir))
    write_json(reports_dir / "p8_qga_calibration_figures_manifest.json", figures)
    selected_text = "No final mask selected yet."
    decision_path = reports_dir / "p8_qga_mask_selection_summary.json"
    if decision_path.exists():
        decision = load_json(decision_path)
        selected_text = (
            f"Selected mask `{decision['selected_mask_id']}` from profile `{decision['profile']}` "
            f"with engineering score `{decision['engineering_score']}`."
        )
    doc = f"""# P8.1.5 — QGA Calibration and Robustness Study

## 1. Objective

Test several QGA profiles, seeds, masks, and short true-Flower validation scenarios before freezing the final L1 mask.

## 2. Why One Mask Is Not Enough

QGA is stochastic, and one mask can reflect one seed and one fitness weighting. Calibration reduces the risk of selecting a brittle feature subset.

## 3. Critical QGA Parameters

- Fitness weights
- FPR penalty
- Min/max feature bounds
- Population size and generations
- Random seed

## 4. Profile Sweep

Profile sweep rows: {len(sweep)}

## 5. Flower 5-Round Validation

Short validation rows: {len(short)}

All ranking runs are validation-only and must not use the global test holdout.

## 6. Final Mask

{selected_text}

## 7. Figures

{chr(10).join(f'- `{path}`' for path in figures)}

## 8. Conclusion

P8.1.5 is an engineering calibration layer. It does not launch P8-b L2, QIFA, FedTN, Docker, or dashboard work.
"""
    (repo_path(config, None) / config["final_experiment_dir"] / "docs" / "08_1_5_qga_calibration.md").write_text(doc, encoding="utf-8")
    print(f"P8.1.5 calibration report built | figures={len(figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
