"""Aggregate latest P7 HeteroFL summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate P7 HeteroFL summaries")
    parser.add_argument("--run-dir", default="experiments/qi-fl-ids-iot-final/outputs/multitier_heterofl")
    parser.add_argument("--reports-dir", default="experiments/qi-fl-ids-iot-final/outputs/reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    run_root = repo_root / args.run_dir
    reports_dir = repo_root / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in run_root.glob("*/alpha_*/k*/latest_run_summary.json")]
    rows = []
    for summary in summaries:
        rows.append(
            {
                "task": summary["task"],
                "alpha": summary["scenario"]["alpha"],
                "clients": summary["scenario"]["clients"],
                "rounds": summary["scenario"]["rounds"],
                "best_round": summary["training"]["best_round"],
                "macro_f1": summary["test"]["metrics"].get("macro_f1", 0.0),
                "accuracy": summary["test"]["metrics"].get("accuracy", 0.0),
                "communication_total_bytes": summary["communication"]["total_bytes"],
                "run_id": summary["run_id"],
                "accepted": summary["accepted"],
            }
        )
    json_path = reports_dir / "p7_multitier_summary.json"
    csv_path = reports_dir / "p7_multitier_summary.csv"
    json_path.write_text(json.dumps({"count": len(rows), "runs": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    print(f"Wrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
