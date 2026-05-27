from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FALLBACK_SELECTED_INDICES = [0, 2, 3, 4, 6, 13, 14, 19, 20, 25, 26, 27]
FALLBACK_MASK_ID = "conservative_seed_42"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_scaler_path() -> Path:
    return (
        repo_root()
        / "experiments"
        / "qi-fl-ids-iot-final"
        / "outputs"
        / "artifacts"
        / "scalers"
        / "l1_binary_robust_scaler.pkl"
    )


def default_mask_path() -> Path:
    return (
        repo_root()
        / "experiments"
        / "qi-fl-ids-iot-final"
        / "outputs"
        / "qga_feature_selection"
        / "final_selected_mask"
        / "feature_mask.json"
    )


@dataclass
class OptionalScaler:
    scaler: Any | None = None
    path: Path | None = None
    available: bool = False
    warnings: list[str] = field(default_factory=list)

    def transform_28(self, features_28: list[float], no_scale: bool = False) -> list[float]:
        if len(features_28) != 28:
            raise ValueError(f"expected 28 features, got {len(features_28)}")
        if no_scale:
            return [float(value) for value in features_28]
        if not self.available or self.scaler is None:
            self.warnings.append("Scaler unavailable; returning unscaled 28-feature vector.")
            return [float(value) for value in features_28]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            transformed = self.scaler.transform([[float(value) for value in features_28]])
        row = transformed[0].tolist() if hasattr(transformed[0], "tolist") else list(transformed[0])
        return [float(value) for value in row]

    def describe(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "path": str(self.path) if self.path else None,
            "warnings": list(dict.fromkeys(self.warnings)),
        }


def load_optional_scaler(path: str | Path | None = None) -> OptionalScaler:
    target = Path(path) if path else default_scaler_path()
    if not target.exists():
        return OptionalScaler(path=target, warnings=[f"Scaler file not found: {target}"])
    try:
        try:
            import joblib

            scaler = joblib.load(target)
        except ImportError:
            with target.open("rb") as handle:
                scaler = pickle.load(handle)
        return OptionalScaler(scaler=scaler, path=target, available=True)
    except Exception as exc:
        return OptionalScaler(path=target, warnings=[f"Scaler could not be loaded: {exc}"])


def scale_28_features(
    features_28: list[float],
    scaler: OptionalScaler | None = None,
    *,
    no_scale: bool = False,
) -> list[float]:
    runtime = scaler or load_optional_scaler()
    return runtime.transform_28(features_28, no_scale=no_scale)


def load_qga_mask(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else default_mask_path()
    if not target.exists():
        return {
            "mask_id": FALLBACK_MASK_ID,
            "selected_indices": list(FALLBACK_SELECTED_INDICES),
            "warnings": [f"QGA mask file not found: {target}"],
        }
    payload = json.loads(target.read_text(encoding="utf-8"))
    indices = payload.get("selected_indices")
    if not isinstance(indices, list) or len(indices) != 12:
        indices = list(FALLBACK_SELECTED_INDICES)
    return {
        "mask_id": str(payload.get("mask_id", FALLBACK_MASK_ID)),
        "selected_indices": [int(index) for index in indices],
        "warnings": [],
    }


def apply_qga_mask(features_28_scaled: list[float], mask_path: str | Path | None = None) -> list[float]:
    if len(features_28_scaled) != 28:
        raise ValueError(f"expected 28 features, got {len(features_28_scaled)}")
    mask = load_qga_mask(mask_path)
    return [float(features_28_scaled[index]) for index in mask["selected_indices"]]
