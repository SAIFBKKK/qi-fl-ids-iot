from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from src.fl.reporting_strategy import ReportingFedAvg
from src.tracking.artifact_logger import build_mlflow_round_metrics


NDArrayList = list[np.ndarray]


@dataclass(frozen=True)
class QIFAClientUpdate:
    """Client update used by the quantum-inspired aggregation kernel."""

    parameters: NDArrayList
    num_examples: int
    node_id: str
    epsilon: float = 0.0


@dataclass(frozen=True)
class QIFAConfig:
    """Configuration for Quantum-Inspired Federated Averaging."""

    lambda_qifa: float = 0.0
    perturbation_enabled: bool = False
    delta_perturbation: float = 0.0
    sigma_noise: float = 0.0
    perturbation_frequency: int = 1
    random_seed: int = 42


def _validate_updates(updates: list[QIFAClientUpdate]) -> None:
    if not updates:
        raise ValueError("QIFA requires at least one client update.")

    reference_shapes = [array.shape for array in updates[0].parameters]
    if any(update.num_examples <= 0 for update in updates):
        raise ValueError("QIFA client num_examples must be positive.")

    for update in updates[1:]:
        shapes = [array.shape for array in update.parameters]
        if shapes != reference_shapes:
            raise ValueError(
                f"QIFA parameter shape mismatch: expected {reference_shapes}, got {shapes}."
            )


def _weighted_average(updates: list[QIFAClientUpdate], weights: np.ndarray) -> NDArrayList:
    return [
        sum(
            float(weight) * np.asarray(update.parameters[layer_idx])
            for weight, update in zip(weights, updates)
        )
        for layer_idx in range(len(updates[0].parameters))
    ]


def _update_norm(arrays: NDArrayList) -> float:
    return float(np.sqrt(sum(float(np.sum(np.square(np.asarray(array)))) for array in arrays)))


def _subtract_arrays(left: NDArrayList, right: NDArrayList) -> NDArrayList:
    return [np.asarray(a) - np.asarray(b) for a, b in zip(left, right)]


def aggregate_qifa_ndarrays(
    updates: list[QIFAClientUpdate],
    *,
    config: QIFAConfig,
    server_round: int,
) -> tuple[NDArrayList, dict[str, float]]:
    """
    Aggregate client parameters with a FedAvg base plus a QI diversity term.

    With lambda_qifa=0 and perturbation disabled, this function returns the exact
    client-weighted FedAvg result.
    """

    _validate_updates(updates)

    examples = np.asarray([update.num_examples for update in updates], dtype=np.float64)
    weights = examples / float(examples.sum())
    fedavg = _weighted_average(updates, weights)

    avg_norm = _update_norm(fedavg)
    epsilons: list[float] = []
    for update in updates:
        delta = _subtract_arrays(update.parameters, fedavg)
        epsilons.append(_update_norm(delta) / (avg_norm + 1e-8))

    effective_weights = weights * (1.0 + float(config.lambda_qifa) * np.asarray(epsilons))
    effective_weight_sum = float(effective_weights.sum())
    if not np.isfinite(effective_weight_sum) or effective_weight_sum <= 0.0:
        effective_weights = weights
    else:
        effective_weights = effective_weights / effective_weight_sum

    aggregated = _weighted_average(updates, effective_weights)
    pre_noise_weight_norm_delta = _update_norm(_subtract_arrays(aggregated, fedavg))

    perturbation_norm = 0.0
    perturbation_applied = 0.0
    if (
        config.perturbation_enabled
        and config.sigma_noise > 0.0
        and config.delta_perturbation != 0.0
        and config.perturbation_frequency > 0
        and int(server_round) % int(config.perturbation_frequency) == 0
    ):
        rng = np.random.default_rng(int(config.random_seed) + int(server_round))
        perturbations: NDArrayList = []
        for layer in aggregated:
            noise = rng.normal(
                loc=0.0,
                scale=float(config.sigma_noise),
                size=layer.shape,
            )
            perturbation = float(config.delta_perturbation) * noise
            perturbations.append(perturbation)
        aggregated = [layer + perturb for layer, perturb in zip(aggregated, perturbations)]
        perturbation_norm = _update_norm(perturbations)
        perturbation_applied = 1.0

    casted = [
        np.asarray(layer, dtype=np.asarray(updates[0].parameters[idx]).dtype)
        for idx, layer in enumerate(aggregated)
    ]
    metrics = {
        "qifa_lambda": float(config.lambda_qifa),
        "qifa_diversity_norm": float(np.mean(epsilons)),
        "qifa_perturbation_norm": float(perturbation_norm),
        "qifa_effective_clients": float(len(updates)),
        "qifa/diversity_mean": float(np.mean(epsilons)),
        "qifa/perturbation_applied": float(perturbation_applied),
        "qifa/weight_norm_delta": float(pre_noise_weight_norm_delta),
    }
    return casted, metrics


class ReportingQIFA(ReportingFedAvg):
    """ReportingFedAvg-compatible strategy with QI diversity adjustment."""

    def __init__(
        self,
        *,
        lambda_qifa: float = 0.0,
        perturbation_enabled: bool = False,
        delta_perturbation: float = 0.0,
        sigma_noise: float = 0.0,
        perturbation_frequency: int = 1,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.qifa_config = QIFAConfig(
            lambda_qifa=float(lambda_qifa),
            perturbation_enabled=bool(perturbation_enabled),
            delta_perturbation=float(delta_perturbation),
            sigma_noise=float(sigma_noise),
            perturbation_frequency=int(perturbation_frequency),
            random_seed=int(random_seed),
        )

    def _epsilon_for_fit_res(self, fit_res: FitRes) -> tuple[str, float]:
        metrics = fit_res.metrics or {}
        node_id = str(metrics.get("node_id") or metrics.get("client_id") or "unknown")
        return node_id, 0.0

    def _fit_results_to_qifa_updates(
        self,
        results: list[tuple[ClientProxy, FitRes]],
    ) -> list[QIFAClientUpdate]:
        updates: list[QIFAClientUpdate] = []
        for _, fit_res in results:
            node_id, epsilon = self._epsilon_for_fit_res(fit_res)
            updates.append(
                QIFAClientUpdate(
                    parameters=parameters_to_ndarrays(fit_res.parameters),
                    num_examples=int(fit_res.num_examples),
                    node_id=node_id,
                    epsilon=epsilon,
                )
            )
        return updates

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        self._record_node_profiles(results)

        if self._multitier_enabled:
            raise ValueError("QIFA is only supported for the flat FL path, not multitier.")
        if not results:
            return None, {}

        adjusted_results = self._apply_expert_weighting(results)
        aggregated_arrays, qifa_metrics = aggregate_qifa_ndarrays(
            self._fit_results_to_qifa_updates(adjusted_results),
            config=self.qifa_config,
            server_round=server_round,
        )
        parameters_aggregated = ndarrays_to_parameters(aggregated_arrays)

        aggregation_fn = getattr(self, "fit_metrics_aggregation_fn", None)
        if aggregation_fn is not None:
            metrics_aggregated: dict[str, Scalar] = aggregation_fn(
                [(fit_res.num_examples, fit_res.metrics or {}) for _, fit_res in results]
            )
        else:
            metrics_aggregated = {}
        metrics_aggregated.update(qifa_metrics)

        if self.tracker is not None:
            self.tracker.record_fit_round(server_round, metrics_aggregated)
        if self.round_metric_logger is not None:
            self.round_metric_logger(
                server_round,
                build_mlflow_round_metrics(metrics_aggregated),
            )

        self._latest_params = parameters_aggregated
        return parameters_aggregated, metrics_aggregated
