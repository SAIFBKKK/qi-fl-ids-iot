"""Run a light P8-b QGA L2 true Flower smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga_l2.config import load_config
from qga_l2.flower_runtime import run_smoke_subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-samples-per-client", type=int, default=1000)
    parser.add_argument("--address", default=None)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--mask-path", default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    summary = run_smoke_subprocess(
        config_path=args.config,
        alpha=args.alpha,
        clients=args.clients,
        rounds=args.rounds,
        max_samples_per_client=args.max_samples_per_client,
        address=args.address or config["flower"]["address"],
        timeout_sec=args.timeout_sec,
        mask_path=args.mask_path,
        evaluate_test=args.evaluate_test,
    )
    print(f"P8-b QGA L2 Flower smoke completed | accepted={summary['accepted']} | run_id={summary['run_id']}")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
