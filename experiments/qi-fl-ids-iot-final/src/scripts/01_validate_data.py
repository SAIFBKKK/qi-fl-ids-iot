"""CLI entry point for P1 final dataset validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.validation import validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final CIC-IoT parquet dataset.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to P1 data validation YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate(args.config)

    print("P1 data validation complete")
    print(f"accepted: {result.accepted}")
    print(f"errors: {len(result.errors)}")
    for error in result.errors:
        print(f"ERROR: {error}")
    print(f"warnings: {len(result.warnings)}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    print("generated files:")
    for path in result.generated_files:
        print(f"- {path}")
    return 0 if result.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
