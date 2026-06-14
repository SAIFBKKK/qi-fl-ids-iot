"""CLI for P5 FedAvg L1 verify mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify P5 FedAvg L1 setup")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    _bootstrap_src_path()
    from fl_l1.verify_setup import verify_setup

    args = parse_args()
    summary = verify_setup(args.config, write_outputs=True)
    print("P5 FedAvg L1 verify complete")
    print(f"accepted: {summary['accepted']}")
    print("checks:")
    for key, value in summary["checks"].items():
        print(f"- {key}: {value}")
    generated = summary.get("generated_files", [])
    if generated:
        print("generated files:")
        for path in generated:
            print(f"- {path}")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
