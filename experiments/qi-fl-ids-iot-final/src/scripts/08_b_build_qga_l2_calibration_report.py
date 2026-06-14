"""Build P8-b QGA L2 calibration report and lightweight figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qga_l2.config import load_config, repo_path, write_json


def _read(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _bar(path: Path, rows: list[dict], key: str, title: str) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["mask_id"] for row in rows]
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#2b8cbe")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    reports = repo_path(config, "outputs.reports_dir")
    figures = repo_path(config, "outputs.figures_dir") / "calibration"
    sweep = _read(reports / "p8b_qga_l2_profile_sweep_summary.csv")
    short = _read(reports / "p8b_qga_l2_flower_short_validation.csv")
    figure_paths: list[str] = []
    specs = [
        ("qga_l2_profiles_macro_f1.png", sweep, "validation_macro_f1", "QGA L2 profile Macro-F1"),
        ("qga_l2_profiles_features_count.png", sweep, "features_count", "QGA L2 selected features"),
        ("qga_l2_profiles_macro_fpr.png", sweep, "validation_macro_fpr", "QGA L2 Macro-FPR"),
        ("qga_l2_profile_fitness_boxplot.png", sweep, "fitness", "QGA L2 fitness"),
        ("qga_l2_short_flower_macro_f1_by_scenario.png", short, "val_macro_f1", "Flower short validation Macro-F1"),
        ("qga_l2_short_flower_fpr_by_scenario.png", short, "val_macro_fpr", "Flower short validation Macro-FPR"),
        ("qga_l2_mask_stability_heatmap.png", sweep, "validation_macro_f1", "QGA L2 mask stability proxy"),
        ("qga_l2_engineering_score_ranking.png", short, "val_macro_f1", "QGA L2 engineering ranking proxy"),
    ]
    for name, rows, key, title in specs:
        output = figures / name
        _bar(output, rows, key, title)
        if output.exists():
            figure_paths.append(output.as_posix())
    write_json(reports / "p8b_qga_l2_figures_manifest.json", figure_paths)
    doc = Path(config["final_experiment_dir"]) / "docs" / "08_b_qga_l2_feature_selection.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        "# P8-b — QGA Feature Selection for L2 Family Classification\n\n"
        "P8-b is experimental and applies QGA feature selection to the L2 attack-family task only.\n\n"
        "The L2 fitness is `0.60*MacroF1 + 0.25*MacroRecall - 0.10*MacroFPR - 0.05*FeatureRatio`.\n\n"
        "QGA mask selection uses train/validation only. The L2 global test holdout is reserved for final Flower evaluation.\n\n"
        "L2 FedAvg + QGA final training must use a true Flower runtime with `test_sent_to_clients=false`.\n",
        encoding="utf-8",
    )
    print(f"P8-b QGA L2 calibration report built | figures={len(figure_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
