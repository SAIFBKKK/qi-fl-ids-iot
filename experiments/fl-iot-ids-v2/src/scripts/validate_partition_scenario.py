from __future__ import annotations

import argparse
import json

import numpy as np

from src.common.paths import DATA_DIR, OUTPUTS_DIR
from src.data.analysis.client_distribution_report import build_client_distribution_report
from src.data.analysis.heatmaps import save_presence_heatmap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    scenario = args.scenario

    scenario_dir = DATA_DIR / "processed" / scenario
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario dir not found: {scenario_dir}")

    reports = {}
    client_names = []
    all_classes = set()

    node_dirs = sorted([p for p in scenario_dir.glob("node*") if p.is_dir()])

    for node_dir in node_dirs:
        train_path = node_dir / "train_preprocessed.npz"
        report = build_client_distribution_report(train_path)
        reports[node_dir.name] = report
        client_names.append(node_dir.name)
        all_classes.update(report["label_counts"].keys())

    sorted_classes = sorted(int(c) for c in all_classes)
    heatmap = np.zeros((len(client_names), len(sorted_classes)), dtype=np.int32)

    for i, client_name in enumerate(client_names):
        counts = reports[client_name]["label_counts"]
        for j, cls in enumerate(sorted_classes):
            heatmap[i, j] = 1 if str(cls) in map(str, counts.keys()) or cls in counts else 0

    report_dir = OUTPUTS_DIR / "reports" / scenario
    figure_dir = OUTPUTS_DIR / "figures" / scenario
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    with (report_dir / "client_distribution_reports.json").open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    save_presence_heatmap(
        heatmap,
        figure_dir / "class_presence_heatmap.png",
        title=f"Class Presence — {scenario}",
    )

    print(f"Validation completed for scenario={scenario}")
    print(f"Reports saved to: {report_dir}")
    print(f"Figures saved to: {figure_dir}")


if __name__ == "__main__":
    main()
