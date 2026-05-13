"""Optional evaluation hooks for P11.

The code-ready phase focuses on structural dry-runs. Full checkpoint-based
evaluation can be added by passing a local checkpoint path in a later run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def checkpoint_available(path: str | Path | None) -> bool:
    if not path:
        return False
    return Path(path).exists() and Path(path).suffix in {".pth", ".pt"}


def evaluation_warning(checkpoint_path: str | Path | None) -> str:
    if checkpoint_available(checkpoint_path):
        return ""
    return "checkpoint_absent_evaluation_skipped"
