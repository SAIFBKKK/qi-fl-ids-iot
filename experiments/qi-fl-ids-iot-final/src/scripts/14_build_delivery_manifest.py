from __future__ import annotations

import importlib.util
from pathlib import Path


FINAL_DIR = Path(__file__).resolve().parents[2]
BUILDER_PATH = FINAL_DIR / "deployment" / "l1_final" / "build_deployment_bundle.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("p14_l1_bundle_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load bundle builder: {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
