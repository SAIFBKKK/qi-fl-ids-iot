from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EdgeInferenceRuntime:
    model_path: str | Path | None = None
    enabled: bool = False

    def predict(self, features: list[float]) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "prediction": None,
                "reason": "edge inference is reserved for a later live-lab step",
                "feature_count": len(features),
            }
        return {
            "enabled": True,
            "prediction": None,
            "reason": "model runtime is not packaged in P16.7",
            "feature_count": len(features),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_path": str(self.model_path) if self.model_path else None,
            "status": "placeholder",
        }
