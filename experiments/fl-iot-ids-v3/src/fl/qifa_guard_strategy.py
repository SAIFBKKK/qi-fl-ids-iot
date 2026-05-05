from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from src.fl.qifa_strategy import (
    NDArrayList,
    _subtract_arrays,
    _update_norm,
    _validate_updates,
    _weighted_average,
)
from src.fl.reporting_strategy import ReportingFedAvg
from src.tracking.artifact_logger import build_mlflow_round_metrics


@dataclass(frozen=True)
class QIFAGuardClientUpdate:
    """Client update plus optional quality signals for QIFA-Guard."""

    parameters: NDArrayList
    num_examples: int
    node_id: str
    global_val_loss: float | None = None
    global_rare_recall: float | None = None


@dataclass(frozen=True)
class QIFAGuardConfig:
    lambda_qifa: float = 0.0
    beta_loss: float = 0.0
    rho_rare: float = 0.0
    min_client_weight: float | None = None
    max_client_weight: float | None = None
    use_global_val_quality: bool = True
    use_rare_bonus: bool = True
    perturbation_enabled: bool = False
    random_seed: int = 42


def _as_qifa_updates(updates: list[QIFAGuardClientUpdate]):
    from src.fl.qifa_strategy import QIFAClientUpdate

    return [
        QIFAClientUpdate(
            parameters=update.parameters,
            num_examples=update.num_examples,
            node_id=update.node_id,
        )
        for update in updates
    ]


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    return None


def _quality_term(update: QIFAGuardClientUpdate, config: QIFAGuardConfig) -> float:
    loss = _finite_or_none(update.global_val_loss)
    if not config.use_global_val_quality or loss is None:
        return 1.0
    exponent = float(np.clip(-float(config.beta_loss) * loss, -700.0, 700.0))
    return float(np.exp(exponent))


def _rare_term(update: QIFAGuardClientUpdate, config: QIFAGuardConfig) -> float:
    recall = _finite_or_none(update.global_rare_recall)
    if not config.use_rare_bonus or recall is None:
        return 1.0
    return float(1.0 + float(config.rho_rare) * recall)


def _clip_scores(scores: np.ndarray, config: QIFAGuardConfig) -> np.ndarray:
    lower = config.min_client_weight
    upper = config.max_client_weight
    if lower is None and upper is None:
        return scores
    low = -np.inf if lower is None else float(lower)
    high = np.inf if upper is None else float(upper)
    return np.clip(scores, low, high)


def _normalize_with_bounds(
    scores: np.ndarray,
    *,
    min_client_weight: float | None,
    max_client_weight: float | None,
) -> np.ndarray:
    if min_client_weight is None and max_client_weight is None:
        total = float(scores.sum())
        if not np.isfinite(total) or total <= 0.0:
            return np.full_like(scores, 1.0 / len(scores), dtype=np.float64)
        return scores / total

    n_clients = len(scores)
    lower = 0.0 if min_client_weight is None else float(min_client_weight)
    upper = 1.0 if max_client_weight is None else float(max_client_weight)
    if lower * n_clients > 1.0 + 1e-12:
        raise ValueError("min_client_weight is infeasible for the number of clients.")
    if upper * n_clients < 1.0 - 1e-12:
        raise ValueError("max_client_weight is infeasible for the number of clients.")

    weights = np.zeros(n_clients, dtype=np.float64)
    fixed = np.zeros(n_clients, dtype=bool)
    remaining = 1.0

    while True:
        free_indices = np.where(~fixed)[0]
        if free_indices.size == 0:
            break
        free_scores = scores[free_indices].astype(np.float64)
        score_sum = float(free_scores.sum())
        if not np.isfinite(score_sum) or score_sum <= 0.0:
            candidate = np.full(free_indices.size, remaining / free_indices.size)
        else:
            candidate = remaining * free_scores / score_sum

        too_low = candidate < lower - 1e-12
        too_high = candidate > upper + 1e-12
        if not np.any(too_low | too_high):
            weights[free_indices] = candidate
            break

        for local_idx, is_low, is_high in zip(free_indices, too_low, too_high):
            if is_low:
                weights[local_idx] = lower
                fixed[local_idx] = True
            elif is_high:
                weights[local_idx] = upper
                fixed[local_idx] = True
        remaining = 1.0 - float(weights[fixed].sum())
        if remaining < -1e-12:
            raise ValueError("Client weight bounds are infeasible.")
        remaining = max(remaining, 0.0)

    total = float(weights.sum())
    if not np.isclose(total, 1.0):
        weights = weights / total
    return weights


def compute_qifa_guard_client_weights(
    updates: list[QIFAGuardClientUpdate],
    *,
    config: QIFAGuardConfig,
) -> tuple[NDArrayList, np.ndarray, np.ndarray, np.ndarray]:
    """Return FedAvg arrays, raw weights, QIFA epsilons, and guard weights."""

    _validate_updates(_as_qifa_updates(updates))
    examples = np.asarray([update.num_examples for update in updates], dtype=np.float64)
    raw_weights = examples / float(examples.sum())
    fedavg = _weighted_average(_as_qifa_updates(updates), raw_weights)

    avg_norm = _update_norm(fedavg)
    epsilons = np.asarray(
        [
            _update_norm(_subtract_arrays(update.parameters, fedavg)) / (avg_norm + 1e-8)
            for update in updates
        ],
        dtype=np.float64,
    )
    quality = np.asarray([_quality_term(update, config) for update in updates], dtype=np.float64)
    rare = np.asarray([_rare_term(update, config) for update in updates], dtype=np.float64)

    scores = raw_weights * (1.0 + float(config.lambda_qifa) * epsilons) * quality * rare
    scores = _clip_scores(scores, config)
    if not np.all(np.isfinite(scores)) or float(scores.sum()) <= 0.0:
        scores = raw_weights
    effective_weights = _normalize_with_bounds(
        scores.astype(np.float64),
        min_client_weight=config.min_client_weight,
        max_client_weight=config.max_client_weight,
    )
    return fedavg, raw_weights, epsilons, effective_weights


def aggregate_qifa_guard_ndarrays(
    updates: list[QIFAGuardClientUpdate],
    *,
    config: QIFAGuardConfig,
    server_round: int,
) -> tuple[NDArrayList, dict[str, float]]:
    fedavg, raw_weights, epsilons, effective_weights = compute_qifa_guard_client_weights(
        updates,
        config=config,
    )
    aggregated = _weighted_average(_as_qifa_updates(updates), effective_weights)
    casted = [
        np.asarray(layer, dtype=np.asarray(updates[0].parameters[idx]).dtype)
        for idx, layer in enumerate(aggregated)
    ]
    metrics = {
        "qifa_guard_lambda": float(config.lambda_qifa),
        "qifa_guard_beta_loss": float(config.beta_loss),
        "qifa_guard_rho_rare": float(config.rho_rare),
        "qifa_guard_effective_clients": float(len(updates)),
        "qifa_guard_diversity_mean": float(np.mean(epsilons)),
        "qifa_guard_weight_norm_delta": float(_update_norm(_subtract_arrays(aggregated, fedavg))),
    }
    for idx, update in enumerate(updates):
        suffix = str(update.node_id).replace("/", "_")
        metrics[f"qifa_guard/raw_weight_{suffix}"] = float(raw_weights[idx])
        metrics[f"qifa_guard/epsilon_{suffix}"] = float(epsilons[idx])
        metrics[f"qifa_guard/effective_weight_{suffix}"] = float(effective_weights[idx])
    metrics["qifa_guard_server_round"] = float(server_round)
    return casted, metrics


class ReportingQIFAGuard(ReportingFedAvg):
    """QIFA extension with validation-quality and rare-class weighting."""

    def __init__(
        self,
        *,
        lambda_qifa: float = 0.0,
        beta_loss: float = 0.0,
        rho_rare: float = 0.0,
        min_client_weight: float | None = None,
        max_client_weight: float | None = None,
        use_global_val_quality: bool = True,
        use_rare_bonus: bool = True,
        perturbation_enabled: bool = False,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.qifa_guard_config = QIFAGuardConfig(
            lambda_qifa=float(lambda_qifa),
            beta_loss=float(beta_loss),
            rho_rare=float(rho_rare),
            min_client_weight=(
                None if min_client_weight is None else float(min_client_weight)
            ),
            max_client_weight=(
                None if max_client_weight is None else float(max_client_weight)
            ),
            use_global_val_quality=bool(use_global_val_quality),
            use_rare_bonus=bool(use_rare_bonus),
            perturbation_enabled=bool(perturbation_enabled),
            random_seed=int(random_seed),
        )

    def _fit_results_to_guard_updates(
        self,
        results: list[tuple[ClientProxy, FitRes]],
    ) -> list[QIFAGuardClientUpdate]:
        updates: list[QIFAGuardClientUpdate] = []
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics or {}
            node_id = self._client_id_from_metrics(client_proxy, metrics)
            updates.append(
                QIFAGuardClientUpdate(
                    parameters=parameters_to_ndarrays(fit_res.parameters),
                    num_examples=int(fit_res.num_examples),
                    node_id=node_id,
                    global_val_loss=_finite_or_none(metrics.get("global_val_loss")),
                    global_rare_recall=_finite_or_none(metrics.get("global_rare_recall")),
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
            raise ValueError("QIFA-Guard is only supported for the flat FL path, not multitier.")
        if not results:
            return None, {}

        adjusted_results = self._apply_expert_weighting(results)
        updates = self._fit_results_to_guard_updates(adjusted_results)
        aggregated_arrays, guard_metrics = aggregate_qifa_guard_ndarrays(
            updates,
            config=self.qifa_guard_config,
            server_round=server_round,
        )
        parameters_aggregated = ndarrays_to_parameters(aggregated_arrays)
        if self._diagnostics_enabled():
            _, _, epsilons, effective_weights = compute_qifa_guard_client_weights(
                updates,
                config=self.qifa_guard_config,
            )
            self._cache_fit_diagnostics(
                server_round,
                adjusted_results,
                parameters_aggregated,
                effective_weights=effective_weights,
                qifa_epsilons=epsilons,
            )

        aggregation_fn = getattr(self, "fit_metrics_aggregation_fn", None)
        if aggregation_fn is not None:
            metrics_aggregated: dict[str, Scalar] = aggregation_fn(
                [(fit_res.num_examples, fit_res.metrics or {}) for _, fit_res in results]
            )
        else:
            metrics_aggregated = {}
        metrics_aggregated.update(guard_metrics)

        if self.tracker is not None:
            self.tracker.record_fit_round(server_round, metrics_aggregated)
        if self.round_metric_logger is not None:
            self.round_metric_logger(
                server_round,
                build_mlflow_round_metrics(metrics_aggregated),
            )

        self._latest_params = parameters_aggregated
        return parameters_aggregated, metrics_aggregated
