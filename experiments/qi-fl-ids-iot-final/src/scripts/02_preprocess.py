"""CLI entry point for P2 preprocessing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P2 preprocessing")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to configs/preprocessing.yaml",
    )
    return parser.parse_args()


def main() -> int:
    _bootstrap_src_path()
    from data.preprocessing import run_preprocessing

    args = parse_args()
    run = run_preprocessing(args.config)
    print("P2 preprocessing complete")
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
