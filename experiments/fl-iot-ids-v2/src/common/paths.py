from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def ensure_runtime_dirs() -> None:
    for path in [
        OUTPUTS_DIR / "logs",
        OUTPUTS_DIR / "metrics",
        OUTPUTS_DIR / "reports",
        OUTPUTS_DIR / "figures",
        OUTPUTS_DIR / "checkpoints",
        MLRUNS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)