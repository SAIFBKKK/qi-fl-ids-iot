from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from loguru import logger


@dataclass(frozen=True)
class HardwareProfile:
    node_id: str
    cpu_cores: int
    ram_mb: int
    device_type: str
    network_quality: str
    battery_powered: bool
    tier_override: str | None = None


@dataclass(frozen=True)
class RegistrationState:
    assigned_tier: str = "legacy"
    model_version: str = "baseline_fedavg_normal_classweights"
    model_source: str = "local_artifacts"
    status: str = "local_fallback"
    error: str | None = None


def collect_hardware_profile() -> HardwareProfile:
    return HardwareProfile(
        node_id=os.getenv("NODE_ID", "node1"),
        cpu_cores=int(os.cpu_count() or 1),
        ram_mb=_detect_ram_mb(),
        device_type=os.getenv("DEVICE_TYPE", "docker_node"),
        network_quality=os.getenv("NETWORK_QUALITY", "medium"),
        battery_powered=_parse_bool(os.getenv("BATTERY_POWERED", "false")),
        tier_override=os.getenv("NODE_TIER_OVERRIDE") or None,
    )


def register_with_model_server(model_server_url: str | None, profile: HardwareProfile) -> RegistrationState:
    if not model_server_url:
        logger.info("model_server_url_absent_using_local_artifacts", node_id=profile.node_id)
        return RegistrationState()

    try:
        import requests

        response = requests.post(
            model_server_url.rstrip("/") + "/nodes/register",
            json=asdict(profile),
            timeout=5,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        state = RegistrationState(
            assigned_tier=str(payload.get("assigned_tier", "legacy")),
            model_version=str(payload.get("model_version", "placeholder")),
            model_source=str(payload.get("model_source", "local_registry")),
            status=str(payload.get("status", "registered")),
        )
        logger.info(
            "node_registered_with_model_server",
            node_id=profile.node_id,
            assigned_tier=state.assigned_tier,
            model_server_url=model_server_url,
        )
        return state
    except Exception as exc:  # noqa: BLE001 - keep legacy inference alive.
        logger.warning(
            "node_registration_failed_using_local_artifacts",
            node_id=profile.node_id,
            model_server_url=model_server_url,
            error=str(exc),
        )
        return RegistrationState(error=str(exc))


def _detect_ram_mb() -> int:
    env_ram = os.getenv("RAM_MB")
    if env_ram:
        try:
            return int(env_ram)
        except ValueError:
            logger.warning("invalid_ram_mb_env_using_default", value=env_ram)

    try:
        import psutil

        return int(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:  # noqa: BLE001 - psutil is optional at runtime.
        return 1024


def _parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}
