from __future__ import annotations

from typing import Any


def validate_raw_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the outer raw-event structure for the future gateway pipeline.

    TODO(P7.3): implement the full raw Node-RED schema validation contract.
    """
    if not isinstance(payload, dict):
        raise ValueError("raw event payload must be a dictionary")
    return payload
