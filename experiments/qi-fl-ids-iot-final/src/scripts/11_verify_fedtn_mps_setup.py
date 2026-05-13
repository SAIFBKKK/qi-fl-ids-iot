"""Verify P11 FedTN/MPS setup."""

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
    parser.add_argument("--config", type=Path, default=Path("experiments/qi-fl-ids-iot-final/configs/fedtn_mps_l1.yaml"))
    args = parser.parse_args()
    _bootstrap()
    from fedtn_mps.config import load_config
    from fedtn_mps.verify_setup import verify_setup

    summary = verify_setup(load_config(args.config))
    print(f"P11 FedTN/MPS verify accepted={summary['accepted']}")
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}")
    for error in summary["errors"]:
        print(f"ERROR: {error}")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
