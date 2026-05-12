"""Sequential P10 grid launcher placeholder.

This script intentionally requires --confirm to avoid accidental full grids.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap() -> None:
    src = Path(__file__).resolve().parents[1]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/qi-fl-ids-iot-final/configs/robustness_l1.yaml"))
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    _bootstrap()
    from robustness.config import load_config

    config = load_config(args.config)
    if not args.confirm:
        print("P10 robustness grid is manual-only. Re-run with --confirm after choosing the exact scenarios.")
        print(f"methods={config['methods']} attacks={config['attack_types']} rates={config['poison_rates']}")
        return 0
    print("Grid execution is intentionally left to the operator; use 10_run_robustness_scenario.py sequentially.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
