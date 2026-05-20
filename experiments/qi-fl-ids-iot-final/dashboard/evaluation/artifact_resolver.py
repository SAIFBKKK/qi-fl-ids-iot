from __future__ import annotations

import glob
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root() / candidate


def first_existing(candidates: list[str]) -> Path | None:
    matches: list[Path] = []
    for item in candidates:
        pattern = resolve(item)
        if any(char in str(pattern) for char in ["*", "?", "["]):
            matches.extend(Path(path) for path in glob.glob(str(pattern)))
        elif pattern.exists():
            matches.append(pattern)
    existing = [path for path in matches if path.exists()]
    if not existing:
        return None
    return sorted(existing, key=lambda path: path.stat().st_mtime, reverse=True)[0]
