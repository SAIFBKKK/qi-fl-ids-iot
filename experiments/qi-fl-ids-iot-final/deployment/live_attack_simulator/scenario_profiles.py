from __future__ import annotations

from dataclasses import dataclass
from random import Random


ALLOWED_SCENARIOS = (
    "benign",
    "ddos_dos_like",
    "recon_like",
    "web_based_like",
    "brute_force_like",
    "spoofing_like",
    "mirai_like",
)

INPUT_MODE_LENGTHS = {
    "selected_12_scaled": 12,
    "original_28_scaled": 28,
}


@dataclass(frozen=True)
class ScenarioProfile:
    name: str
    description: str
    center: float
    spread: float


SCENARIO_PROFILES = {
    "benign": ScenarioProfile("benign", "Low-amplitude controlled baseline feature replay.", -0.15, 0.08),
    "ddos_dos_like": ScenarioProfile("ddos_dos_like", "High-rate-like scaled feature pattern.", 1.75, 0.22),
    "recon_like": ScenarioProfile("recon_like", "Short-window probing-like scaled feature pattern.", 0.75, 0.16),
    "web_based_like": ScenarioProfile("web_based_like", "Application-heavy scaled feature pattern.", 1.05, 0.14),
    "brute_force_like": ScenarioProfile("brute_force_like", "Repeated-authentication-like scaled feature pattern.", 1.25, 0.18),
    "spoofing_like": ScenarioProfile("spoofing_like", "Identity-change-like scaled feature pattern.", 0.65, 0.13),
    "mirai_like": ScenarioProfile("mirai_like", "Botnet-family-like scaled feature pattern.", 1.55, 0.20),
}


def get_profile(scenario: str) -> ScenarioProfile:
    try:
        return SCENARIO_PROFILES[scenario]
    except KeyError as exc:
        raise ValueError(f"unsupported scenario: {scenario}") from exc


def feature_count_for_mode(input_mode: str) -> int:
    try:
        return INPUT_MODE_LENGTHS[input_mode]
    except KeyError as exc:
        raise ValueError(f"unsupported input_mode: {input_mode}") from exc


def generate_controlled_features(scenario: str, input_mode: str, sequence: int = 0) -> list[float]:
    profile = get_profile(scenario)
    count = feature_count_for_mode(input_mode)
    rng = Random(f"{scenario}:{input_mode}:{sequence}")
    values: list[float] = []
    for index in range(count):
        shape = ((index % 7) - 3) * profile.spread * 0.18
        drift = (sequence % 13) * 0.01
        values.append(round(profile.center + shape + drift + rng.uniform(-profile.spread, profile.spread), 6))
    return values

