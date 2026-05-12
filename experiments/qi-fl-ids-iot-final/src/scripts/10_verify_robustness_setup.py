"""Verify P10 robustness setup."""

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
    args = parser.parse_args()
    _bootstrap()
    from robustness.config import load_config
    from robustness.verify_setup import verify_setup

    summary = verify_setup(load_config(args.config))
    print(f"P10 robustness verify accepted={summary['accepted']}")
    for error in summary["errors"]:
        print(f"ERROR: {error}")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
