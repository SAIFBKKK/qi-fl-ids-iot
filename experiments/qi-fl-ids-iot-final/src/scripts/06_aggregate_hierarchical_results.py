"""Aggregate latest P6 hierarchical summaries if full/smoke runs exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate P6 hierarchical Flower latest summaries")
    parser.add_argument("--run-dir", default="experiments/qi-fl-ids-iot-final/outputs/hierarchical_flower")
    parser.add_argument("--output", default="experiments/qi-fl-ids-iot-final/outputs/reports/hierarchical_flower_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    run_root = repo_root / args.run_dir
    summaries = []
    for path in run_root.glob("*/alpha_*/k*/latest_run_summary.json"):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"count": len(summaries), "summaries": summaries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output} with {len(summaries)} P6 summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
