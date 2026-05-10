"""Build P9 QIFA ablation report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qifa.config import load_config
from qifa.report_builder import build_qifa_ablation_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    rows = build_qifa_ablation_report(config, Path.cwd().resolve())
    print(f"P9 QIFA ablation report built | rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
