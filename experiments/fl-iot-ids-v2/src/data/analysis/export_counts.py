from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def export_label_counts(y: np.ndarray, out_path: str | Path) -> None:
    values, counts = np.unique(y, return_counts=True)
    payload = {int(v): int(c) for v, c in zip(values, counts)}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)