"""Feature mask loading, serialization, and application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from qga.config import load_json


def load_feature_names(path: str | Path) -> list[str]:
    payload = load_json(path)
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict) and "feature_names" in payload:
        return [str(item) for item in payload["feature_names"]]
    raise ValueError(f"unsupported feature_names format: {path}")


def selected_indices(mask: np.ndarray) -> list[int]:
    return [int(idx) for idx in np.flatnonzero(np.asarray(mask, dtype=np.int8) == 1)]


def selected_feature_names(mask: np.ndarray, feature_names: list[str]) -> list[str]:
    return [feature_names[idx] for idx in selected_indices(mask)]


def mask_payload(mask: np.ndarray, feature_names: list[str], *, run_id: str, method: str) -> dict[str, Any]:
    indices = selected_indices(mask)
    return {
        "phase": "P8",
        "method": method,
        "run_id": run_id,
        "n_features_original": len(feature_names),
        "selected_features_count": len(indices),
        "selected_indices": indices,
        "selected_features": [feature_names[idx] for idx in indices],
        "mask": np.asarray(mask, dtype=int).tolist(),
    }


def apply_feature_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    indices = selected_indices(mask)
    if not indices:
        raise ValueError("cannot apply empty feature mask")
    return np.asarray(X)[:, indices]


def load_latest_mask(qga_dir: str | Path) -> dict[str, Any]:
    latest_path = Path(qga_dir) / "latest_run_summary.json"
    if not latest_path.exists():
        raise FileNotFoundError(f"QGA latest summary not found: {latest_path}")
    summary = load_json(latest_path)
    mask_path = summary.get("qga", {}).get("feature_mask_path")
    if not mask_path:
        for artifact in summary.get("artifacts", []):
            if str(artifact).endswith("feature_mask.json"):
                mask_path = artifact
                break
    if not mask_path:
        raise ValueError("latest QGA summary does not reference feature_mask.json")
    payload = load_json(mask_path)
    mask = np.asarray(payload["mask"], dtype=np.int8)
    return {"summary": summary, "payload": payload, "mask": mask}
