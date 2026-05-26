from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EdgeInferenceRuntime:
    model_path: str | None = None
    enabled: bool = False

    def predict(self, features: list[float]) -> dict[str, object]:
        if not self.enabled:
            return {"enabled": False, "prediction": None, "reason": "edge inference placeholder"}
        return {"enabled": True, "feature_count": len(features), "prediction": None}

    def describe(self) -> dict[str, object]:
        return {"enabled": self.enabled, "model_path": self.model_path, "status": "placeholder"}

