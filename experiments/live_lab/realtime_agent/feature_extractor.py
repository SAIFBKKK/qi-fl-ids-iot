from __future__ import annotations

import sys
from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parents[1] / "nodes" / "common"
sys.path.insert(0, str(COMMON_DIR))

from feature_schema import expected_28_features  # noqa: E402
from flow_window import FlowWindow  # noqa: E402


def extract_28_features_from_window(window: FlowWindow) -> dict[str, float | None]:
    """Return an explicit placeholder row; unsupported values remain None."""
    return {feature: None for feature in expected_28_features()} | {"Number": float(len(window.packets))}

