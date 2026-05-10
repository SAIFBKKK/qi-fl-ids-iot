"""Verify P8 QGA setup without training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga.config import load_config
from qga.verify_setup import verify_qga_setup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = verify_qga_setup(config)
    print(f"P8 QGA verify accepted={summary['accepted']}")
    if summary["warnings"]:
        print("Warnings:", "; ".join(summary["warnings"]))
    if summary["errors"]:
        print("Errors:", "; ".join(summary["errors"]))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
