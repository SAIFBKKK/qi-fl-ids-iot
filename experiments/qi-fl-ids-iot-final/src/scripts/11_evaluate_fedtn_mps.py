"""Evaluate a compressed P11 model if a local checkpoint is available."""

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
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    _bootstrap()
    from fedtn_mps.evaluation import evaluation_warning

    warning = evaluation_warning(args.checkpoint)
    if warning:
        print(f"WARNING: {warning}")
        return 0
    print("Checkpoint evaluation hook is ready; metric evaluation is intentionally manual in P11 code-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
