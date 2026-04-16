from __future__ import annotations

from pathlib import Path
import pickle


def load_label_mapping(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label mapping artifact not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)