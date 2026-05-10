"""Run a light true Flower smoke test for P9 QIFA."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qifa.config import load_config
from qifa.data import configured_address
from qifa.flower_runtime import run_qifa_smoke_subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-samples-per-client", type=int, default=1000)
    parser.add_argument("--variant", default="hybrid")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--address", default=None)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--use-qga-mask", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    address = configured_address(config, args.address)
    summary = run_qifa_smoke_subprocess(
        config_path=args.config,
        alpha=float(args.alpha),
        clients=int(args.clients),
        rounds=int(args.rounds),
        variant=str(args.variant),
        gamma=float(args.gamma),
        address=address,
        max_samples_per_client=int(args.max_samples_per_client),
        timeout_sec=int(args.timeout_sec),
        use_qga_mask=bool(args.use_qga_mask),
    )
    print(f"P9 QIFA smoke completed | accepted={summary['accepted']} | run_id={summary['run_id']}")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
