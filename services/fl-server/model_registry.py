from __future__ import annotations

from dataclasses import asdict, dataclass

from tier_assignment import VALID_TIERS


@dataclass(frozen=True)
class ModelMetadata:
    tier: str
    model_version: str
    model_source: str
    status: str
    description: str
    artifact_path: str | None = None


class ModelRegistry:
    def __init__(self) -> None:
        self._models = {
            tier: ModelMetadata(
                tier=tier,
                model_version="placeholder",
                model_source="local_registry",
                status="placeholder",
                description=(
                    f"{tier} IDS model placeholder. Real weak/medium/powerful bundles "
                    "will be attached after offline training completes."
                ),
                artifact_path=f"/artifacts/{tier}",
            )
            for tier in VALID_TIERS
        }

    def list_models(self) -> list[dict]:
        return [asdict(self._models[tier]) for tier in VALID_TIERS]

    def get_metadata(self, tier: str) -> dict:
        normalized = tier.strip().lower()
        if normalized not in self._models:
            raise KeyError(normalized)
        return asdict(self._models[normalized])
