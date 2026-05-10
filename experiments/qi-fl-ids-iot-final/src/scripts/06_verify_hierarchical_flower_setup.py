"""Verify P6 hierarchical Flower setup without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify P6 hierarchical Flower setup")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    _bootstrap_src_path()
    from fl_hierarchical.verify_setup import verify_hierarchical_setup

    args = parse_args()
    summary = verify_hierarchical_setup(args.config, write_outputs=True)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
