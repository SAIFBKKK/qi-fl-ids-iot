"""CLI verify for P5.2 true Flower L1 runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify P5.2 Flower L1 setup")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    _bootstrap_src_path()
    from fl_l1_flower.verify_flower_setup import verify_flower_setup

    args = parse_args()
    summary = verify_flower_setup(args.config, write_outputs=True)
    print("P5.2 Flower L1 verify complete")
    print(f"accepted: {summary['accepted']}")
    print(f"flower_version: {summary['flower_version']}")
    print(f"architecture: {summary['architecture']}")
    print("checks:")
    for key, value in summary["checks"].items():
        print(f"- {key}: {value}")
    if summary.get("warnings"):
        print("warnings:")
        for warning in summary["warnings"]:
            print(f"- {warning}")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

