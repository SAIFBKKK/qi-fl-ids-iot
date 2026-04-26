from __future__ import annotations

from typing import Any

from src.common.paths import ARTIFACTS_DIR
from src.tracking.run_naming import generate_run_name


def configure_runtime_artifacts(
    experiment: dict[str, Any],
    config: dict[str, Any],
) -> str:
    run_name = generate_run_name(experiment)
    runtime_cfg = config.setdefault("runtime", {})
    runtime_cfg["run_name"] = run_name
    runtime_cfg.setdefault(
        "scaffold_state_dir",
        str(ARTIFACTS_DIR / "scaffold_state" / run_name),
    )
    return run_name
