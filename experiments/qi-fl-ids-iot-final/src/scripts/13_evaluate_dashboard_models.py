"""Evaluate or report P13 dashboard L1 models."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path


def _bootstrap() -> None:
    dashboard = Path(__file__).resolve().parents[2] / "dashboard"
    if str(dashboard) not in sys.path:
        sys.path.insert(0, str(dashboard))


def main() -> int:
    _bootstrap()
    from evaluation.evaluator import evaluate_models, write_evaluation_outputs

    rows, warnings = evaluate_models()
    write_evaluation_outputs(rows, warnings)
    builder_path = Path(__file__).with_name("13_build_dashboard_assets.py")
    spec = importlib.util.spec_from_file_location("p13_dashboard_asset_builder", builder_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    print(f"P13 dashboard model evaluation complete models={len(rows)} warnings={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
