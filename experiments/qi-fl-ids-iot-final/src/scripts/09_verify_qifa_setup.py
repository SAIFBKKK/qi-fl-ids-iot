"""Verify P9 QIFA setup without training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qifa.verify_setup import verify_qifa_setup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    summary = verify_qifa_setup(args.config, write_outputs=True)
    print(f"P9 QIFA verify completed | accepted={summary['accepted']} | flower_version={summary['flower_version']}")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
