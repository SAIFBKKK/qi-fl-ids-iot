"""Feature-mask utilities for P8-b QGA L2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from qga_l2.config import load_json, repo_path


def load_feature_names(path: str | Path) -> list[str]:
    payload = load_json(path)
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict) and "feature_names" in payload:
        return [str(item) for item in payload["feature_names"]]
    raise ValueError(f"unsupported feature names format: {path}")


def selected_indices(mask: np.ndarray) -> list[int]:
    return [int(index) for index in np.flatnonzero(np.asarray(mask, dtype=np.int8) == 1)]


def selected_feature_names(mask: np.ndarray, feature_names: list[str]) -> list[str]:
    return [feature_names[index] for index in selected_indices(mask)]


def apply_feature_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    indices = selected_indices(mask)
    if not indices:
        raise ValueError("cannot apply an empty QGA L2 feature mask")
    return np.asarray(X)[:, indices]


def mask_payload(mask: np.ndarray, feature_names: list[str], *, mask_id: str, profile: str, seed: int, method: str) -> dict[str, Any]:
    indices = selected_indices(mask)
    return {
        "phase": "P8-b",
        "method": method,
        "mask_id": mask_id,
        "profile": profile,
        "seed": int(seed),
        "n_features_original": len(feature_names),
        "selected_features_count": len(indices),
        "selected_indices": indices,
        "selected_features": [feature_names[index] for index in indices],
        "mask": np.asarray(mask, dtype=int).tolist(),
    }


def load_final_mask(config: dict[str, Any]) -> dict[str, Any]:
    final_dir = repo_path(config, "outputs.qga_l2_dir") / "final_selected_mask"
    mask_path = final_dir / "feature_mask.json"
    selected_path = final_dir / "selected_features.json"
    decision_path = final_dir / "selection_decision.json"
    missing = [path for path in (mask_path, selected_path, decision_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("P8-b final_selected_mask is incomplete: " + ", ".join(path.as_posix() for path in missing))
    payload = load_json(mask_path)
    decision = load_json(decision_path)
    mask = np.asarray(payload["mask"], dtype=np.int8)
    count = int(mask.sum())
    if int(payload.get("selected_features_count", count)) != count:
        raise ValueError("P8-b final mask feature count mismatch")
    selected_mask_id = str(decision.get("selected_mask_id") or payload.get("mask_id"))
    if payload.get("mask_id") != selected_mask_id:
        raise ValueError("P8-b final mask id mismatch")
    return {
        "payload": {
            **payload,
            "selected_mask_id": selected_mask_id,
            "selected_mask_source": "final_selected_mask",
            "calibration_decision_used": True,
            "feature_mask_path": mask_path.as_posix(),
            "selection_decision_path": decision_path.as_posix(),
        },
        "decision": decision,
        "mask": mask,
    }
