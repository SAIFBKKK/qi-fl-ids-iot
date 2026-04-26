from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.common.logger import get_logger
from src.common.schemas import NodeProfile, TierAssignment


class NodeProfiler:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.tiers = self._load_tiers(self.config_path)
        self._assignments: dict[str, TierAssignment] = {}
        self.logger = get_logger("node_profiler")

    def _load_tiers(self, path: str | Path) -> dict[str, dict[str, Any]]:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Tier profile config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        tiers = payload.get("tiers")
        if not isinstance(tiers, dict):
            raise ValueError(f"Invalid tier config: missing 'tiers' mapping in {config_path}")

        for tier_name in ("weak", "medium", "powerful"):
            if tier_name not in tiers:
                raise ValueError(f"Invalid tier config: missing tier {tier_name!r}")
            tier = tiers[tier_name]
            for field in ("model_width", "local_epochs", "batch_size"):
                if field not in tier:
                    raise ValueError(
                        f"Invalid tier config: missing {field!r} in tier {tier_name!r}"
                    )

        return tiers

    def assign_tier(self, profile: NodeProfile) -> TierAssignment:
        weak_max_ram = int(self.tiers["weak"]["max_ram_mb"])
        medium_max_ram = int(self.tiers["medium"]["max_ram_mb"])
        powerful_device_types = set(self.tiers["powerful"].get("device_types", ()))

        if profile.ram_mb < weak_max_ram or profile.battery_powered:
            tier_name = "weak"
        elif weak_max_ram <= profile.ram_mb < medium_max_ram:
            tier_name = "medium"
        elif profile.ram_mb >= medium_max_ram and profile.device_type in powerful_device_types:
            tier_name = "powerful"
        else:
            tier_name = "medium"

        assignment = self._build_assignment(tier_name)
        self._assignments[profile.node_id] = assignment
        self.logger.info(
            "Assigned node_id=%s tier=%s model_width=%.2f local_epochs=%s batch_size=%s",
            profile.node_id,
            assignment.assigned_tier,
            assignment.model_width,
            assignment.local_epochs,
            assignment.batch_size,
        )
        return assignment

    def get_assignment(self, node_id: str) -> TierAssignment | None:
        return self._assignments.get(node_id)

    def list_assignments(self) -> dict[str, str]:
        return {
            node_id: assignment.assigned_tier
            for node_id, assignment in sorted(self._assignments.items())
        }

    def _build_assignment(self, tier_name: str) -> TierAssignment:
        tier = self.tiers[tier_name]
        return TierAssignment(
            assigned_tier=tier_name,
            model_width=float(tier["model_width"]),
            local_epochs=int(tier["local_epochs"]),
            batch_size=int(tier["batch_size"]),
        )
