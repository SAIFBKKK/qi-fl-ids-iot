from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from src.data.datasets.flat_dataset import load_npz_xy


def build_client_distribution_report(npz_path: str | Path) -> dict:
    x, y = load_npz_xy(npz_path)
    values, counts = np.unique(y, return_counts=True)

    return {
        "file": str(npz_path),
        "num_samples": int(len(y)),
        "num_features": int(x.shape[1]),
        "num_classes_present": int(len(values)),
        "label_counts": {int(v): int(c) for v, c in zip(values, counts)},
    }


def save_client_distribution_report(npz_path: str | Path, out_json: str | Path) -> None:
    report = build_client_distribution_report(npz_path)
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)