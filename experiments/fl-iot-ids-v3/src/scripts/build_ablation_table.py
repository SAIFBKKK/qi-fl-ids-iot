from __future__ import annotations

import csv
import json
from pathlib import Path

from src.common.paths import OUTPUTS_DIR


def build_ablation_table(outputs_dir: str | Path | None = None) -> None:
    reports_root = (
        Path(outputs_dir)
        if outputs_dir is not None
        else OUTPUTS_DIR / "reports" / "baselines"
    )

    metrics_order = [
        ("final_macro_f1", "Macro-F1"),
        ("final_benign_recall", "Benign Recall"),
        ("final_false_positive_rate", "FPR"),
        ("final_rare_class_recall", "Rare Recall"),
        ("final_accuracy", "Accuracy"),
        ("final_recall_macro", "Recall Macro"),
    ]

    rows: list[dict[str, str]] = []
    if not reports_root.exists():
        print(f"No baseline report directory found: {reports_root}")
        return

    for experiment_dir in sorted(reports_root.iterdir()):
        summary_path = experiment_dir / "run_summary.json"
        if not summary_path.exists():
            continue

        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

        if summary.get("status") not in {"success", "partial"}:
            continue

        row = {
            "Experiment": summary.get("experiment_name", experiment_dir.name),
            "Strategy": summary.get("fl_strategy", "-"),
            "Scenario": summary.get("data_scenario", "-"),
            "Imbalance": summary.get("imbalance_strategy", "-"),
            "Rounds": str(summary.get("completed_rounds", "-")),
        }
        for key, label in metrics_order:
            value = summary.get(key)
            row[label] = f"{value:.4f}" if isinstance(value, (int, float)) else "-"
        rows.append(row)

    if not rows:
        print("No completed baseline runs found.")
        return

    headers = list(rows[0].keys())
    markdown_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        markdown_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

    reports_root.parent.mkdir(parents=True, exist_ok=True)
    csv_path = reports_root.parent / "fl_v3_ablation_table.csv"
    md_path = reports_root.parent / "fl_v3_ablation_table.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown_lines) + "\n")

    print(f"Ablation table exported to {csv_path}")
    print(f"Markdown version exported to {md_path}")


if __name__ == "__main__":
    build_ablation_table()
