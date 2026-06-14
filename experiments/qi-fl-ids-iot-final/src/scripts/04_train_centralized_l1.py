"""CLI entry point for P4 centralized L1 binary baseline training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P4 centralized L1 baseline")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to configs/centralized_l1.yaml",
    )
    return parser.parse_args()


def main() -> int:
    _bootstrap_src_path()
    from models.training import train_centralized_l1

    args = parse_args()
    run = train_centralized_l1(args.config)
    print("P4 centralized L1 training complete")
    print(f"accepted: {run.accepted}")
    print(f"errors: {len(run.errors)}")
    for error in run.errors:
        print(f"ERROR: {error}")
    print(f"warnings: {len(run.warnings)}")
    for warning in run.warnings:
        print(f"WARNING: {warning}")
    print("generated files:")
    for path in run.generated_files:
        print(f"- {path}")
    return 0 if run.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
