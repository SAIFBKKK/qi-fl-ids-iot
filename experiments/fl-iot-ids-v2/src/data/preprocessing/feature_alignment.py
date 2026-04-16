from __future__ import annotations

from pathlib import Path
import pickle


def load_feature_names(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature names artifact not found: {path}")
    with path.open("rb") as f:
        feature_names = pickle.load(f)
    return list(feature_names)


def validate_feature_count(feature_names: list[str], expected_count: int) -> None:
    if len(feature_names) != expected_count:
        raise ValueError(
            f"Feature count mismatch: expected={expected_count}, got={len(feature_names)}"
        )