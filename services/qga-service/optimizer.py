from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class OptimizationInput:
    available_features: int
    latency_budget_ms: float
    energy_budget: float
    risk_tolerance: float


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    selected_features: list[str]
    feature_budget: int
    threshold_suggestion: float
    optimization_score: float
    qga_iterations: int
    mode: str


class DeterministicQGAOptimizer:
    """Lightweight deterministic stub for the future QGA optimizer.

    The service is intentionally not a full QGA implementation in P6C. It maps
    simple IoT constraints to a repeatable recommendation so the microservice
    contract, observability, and Docker profile can be validated.
    """

    def __init__(self, default_iterations: int) -> None:
        self.default_iterations = max(1, default_iterations)

    def optimize(self, request: OptimizationInput) -> OptimizationResult:
        available_features = max(1, request.available_features)
        latency_factor = _clamp(request.latency_budget_ms / 10.0, 0.0, 1.0)
        energy_factor = _clamp(request.energy_budget, 0.0, 1.0)
        risk_factor = _clamp(request.risk_tolerance, 0.0, 1.0)

        budget_ratio = _clamp(
            0.25 + (0.35 * energy_factor) + (0.25 * latency_factor) - (0.10 * risk_factor),
            0.15,
            0.95,
        )
        feature_budget = round(available_features * budget_ratio)
        feature_budget = max(1, min(available_features, feature_budget))

        threshold = _clamp(
            0.50 + (0.20 * (1.0 - risk_factor)) - (0.08 * (1.0 - energy_factor)),
            0.25,
            0.90,
        )
        score = _clamp(
            0.50
            + (0.20 * energy_factor)
            + (0.15 * latency_factor)
            + (0.10 * (1.0 - risk_factor))
            + (0.05 * (1.0 - (feature_budget / available_features))),
            0.0,
            1.0,
        )

        selected_features = [
            f"feature_{index:02d}" for index in range(1, feature_budget + 1)
        ]

        return OptimizationResult(
            status="ok",
            selected_features=selected_features,
            feature_budget=feature_budget,
            threshold_suggestion=round(threshold, 4),
            optimization_score=round(score, 4),
            qga_iterations=self.default_iterations,
            mode="deterministic_stub",
        )
