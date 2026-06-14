"""Runtime wrapper for the scientific FL runner.

This module patches the scientific runner's tracking URI resolution without
modifying the mounted experiment source tree.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

SCIENTIFIC_ROOT = Path(
    os.environ.get("REAL_FL_WORKDIR", "/app/experiments/fl-iot-ids-v3")
)
if str(SCIENTIFIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SCIENTIFIC_ROOT))

from src.scripts import run_experiment as runner_module  # noqa: E402


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _patched_resolve_tracking_uri(raw_uri: str) -> str:
    if raw_uri.startswith(("http://", "https://")):
        patched_uri = raw_uri
    else:
        patched_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    logger.info(
        "[wrapper] resolve_tracking_uri monkey-patch: raw_uri=%s -> patched=%s",
        raw_uri,
        patched_uri,
    )
    return patched_uri


runner_module.resolve_tracking_uri = _patched_resolve_tracking_uri
logger.info("[wrapper] monkey-patch applied on runner_module.resolve_tracking_uri")


if __name__ == "__main__":
    runner_module.main()
