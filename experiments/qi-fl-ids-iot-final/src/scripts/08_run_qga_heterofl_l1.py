"""Run HeteroFL L1 with the latest QGA mask."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga.config import load_config
from qga.heterofl_adapter import run_qga_heterofl_l1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-samples-per-client", type=int, default=1000)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = run_qga_heterofl_l1(
        config=config,
        mode=args.mode,
        alpha=args.alpha,
        clients=args.clients,
        rounds=args.rounds,
        max_samples_per_client=args.max_samples_per_client,
    )
    print(f"P8 HeteroFL+QGA completed | accepted={summary['accepted']} | run_id={summary['run_id']}")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
