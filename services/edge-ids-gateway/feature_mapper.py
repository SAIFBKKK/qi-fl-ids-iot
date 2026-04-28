from __future__ import annotations

from typing import Any


def map_raw_to_features(raw_event: dict[str, Any]) -> dict[str, float]:
    """Convert a validated raw event into the 28-feature IDS vector.

    TODO(P7.4): implement deterministic raw-event to CIC-IoT feature mapping.
    """
    raise NotImplementedError("raw-to-feature mapping is not implemented in P7.2")
