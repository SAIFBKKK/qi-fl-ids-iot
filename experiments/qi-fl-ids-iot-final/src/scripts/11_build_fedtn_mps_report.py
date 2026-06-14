"""Build P11 FedTN/MPS reports."""

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
    from fedtn_mps.report_builder import build_reports

    result = build_reports(load_config(args.config))
    print(f"P11 FedTN/MPS report built rows={result['rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
