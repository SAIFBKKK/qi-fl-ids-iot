from __future__ import annotations

import csv
import json
from pathlib import Path


def build_ablation_table(outputs_dir: str = "outputs/reports/baselines") -> None:
    metrics_order = [
        ("final_macro_f1",            "Macro-F1"),
        ("final_benign_recall",       "Benign Recall"),
        ("final_false_positive_rate", "FPR ↓"),
        ("final_rare_class_recall",   "Rare Recall"),
        ("final_accuracy",            "Accuracy"),
        ("final_recall_macro",        "Recall Macro"),
    ]

    rows = []
    for exp_dir in sorted(Path(outputs_dir).iterdir()):
        summary_path = exp_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            d = json.load(f)
        if d.get("status") != "success":
            continue
        row = {
            "Experiment": d.get("experiment_name", exp_dir.name),
            "Strategy":   d.get("fl_strategy", "-"),
            "Scenario":   d.get("data_scenario", "-"),
            "Imbalance":  d.get("imbalance_strategy", "-"),
            "Rounds":     d.get("completed_rounds", "-"),
        }
        for key, label in metrics_order:
            val = d.get(key)
            row[label] = f"{val:.4f}" if val is not None else "-"
        rows.append(row)

    if not rows:
        print("Aucun résultat trouvé.")
        return

    cols = list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    print(header)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(row[c]) for c in cols) + " |")

    out_path = Path(outputs_dir) / "ablation_table.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nTable exportée → {out_path}")


if __name__ == "__main__":
    build_ablation_table()
