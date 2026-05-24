from __future__ import annotations

from datetime import UTC, datetime
from random import Random


INPUT_MODE_LENGTHS = {
    "selected_12_scaled": 12,
    "original_28_scaled": 28,
}

SCENARIO_OFFSETS = {
    "benign": 0.0,
    "ddos_dos_like": 1.8,
    "recon_like": 0.9,
    "web_based_like": 1.1,
    "brute_force_like": 1.3,
    "spoofing_like": 0.7,
    "mirai_like": 1.6,
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def feature_count_for_mode(input_mode: str) -> int:
    try:
        return INPUT_MODE_LENGTHS[input_mode]
    except KeyError as exc:
        raise ValueError(f"unsupported input_mode: {input_mode}") from exc


def synthetic_feature_vector(input_mode: str, scenario: str = "benign", sequence: int = 0) -> list[float]:
    count = feature_count_for_mode(input_mode)
    offset = SCENARIO_OFFSETS.get(scenario, 0.0)
    rng = Random(f"{input_mode}:{scenario}:{sequence}")
    values: list[float] = []
    for index in range(count):
        wave = ((index % 5) - 2) * 0.08
        drift = (sequence % 11) * 0.015
        values.append(round(offset + wave + drift + rng.uniform(-0.03, 0.03), 6))
    return values


def build_flow_payload(
    node_id: str,
    input_mode: str,
    scenario: str = "benign",
    sequence: int = 0,
    features: list[float] | None = None,
) -> dict[str, object]:
    vector = features if features is not None else synthetic_feature_vector(input_mode, scenario, sequence)
    expected_count = feature_count_for_mode(input_mode)
    if len(vector) != expected_count:
        raise ValueError(f"{input_mode} expects {expected_count} features, got {len(vector)}")
    return {
        "schema_version": "1.0",
        "flow_id": f"{node_id}-{input_mode}-{scenario}-{sequence:06d}",
        "node_id": node_id,
        "timestamp": utc_now(),
        "input_mode": input_mode,
        "scenario": scenario,
        "features": [float(value) for value in vector],
    }

