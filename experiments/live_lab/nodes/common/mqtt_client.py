from __future__ import annotations

from typing import Any

from utils import utc_now


def build_topic(node_id: str, kind: str = "flows") -> str:
    allowed = {"flows", "predictions", "alerts", "status"}
    if kind not in allowed:
        raise ValueError(f"unsupported topic kind: {kind}")
    return f"ids/{kind}/{node_id}"


def build_payload(
    node_id: str,
    input_mode: str,
    features: list[float],
    scenario: str = "dry_run",
    flow_id: str | None = None,
) -> dict[str, Any]:
    return {
        "flow_id": flow_id or f"{node_id}-dry-run-flow",
        "node_id": node_id,
        "timestamp": utc_now(),
        "input_mode": input_mode,
        "scenario": scenario,
        "features": [float(value) for value in features],
    }

