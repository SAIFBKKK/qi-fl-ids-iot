from __future__ import annotations

import math
import pickle
from typing import Any

import numpy as np
from flwr.common import EvaluateRes, FitIns, FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.tracking.artifact_logger import BaselineArtifactTracker


class ReportingFedAvg(FedAvg):
    def __init__(
        self,
        *,
        tracker: BaselineArtifactTracker | None = None,
        monitor_metric: str = "macro_f1",
        expert_node_id: str | None = None,
        expert_factor: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tracker = tracker
        self.monitor_metric = monitor_metric
        self.expert_node_id = expert_node_id
        self.expert_factor = expert_factor
        self._best_metric: float = -math.inf
        self._best_round: int = 0
        self._best_params: Parameters | None = None
        self._latest_params: Parameters | None = None

    @property
    def best_round_info(self) -> dict[str, object]:
        return {
            "best_round": self._best_round,
            "best_metric": self._best_metric,
            "monitor_metric": self.monitor_metric,
        }

    def _apply_expert_weighting(
        self,
        results: list[tuple[ClientProxy, FitRes]],
    ) -> list[tuple[ClientProxy, FitRes]]:
        if not self.expert_node_id or self.expert_factor == 1.0:
            return results

        adjusted: list[tuple[ClientProxy, FitRes]] = []
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics or {}
            node_id = metrics.get("node_id") or metrics.get("client_id")
            if node_id == self.expert_node_id:
                fit_res = FitRes(
                    status=fit_res.status,
                    parameters=fit_res.parameters,
                    num_examples=int(fit_res.num_examples * self.expert_factor),
                    metrics=fit_res.metrics,
                )
            adjusted.append((client_proxy, fit_res))
        return adjusted

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        adjusted_results = self._apply_expert_weighting(results)
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round,
            results=adjusted_results,
            failures=failures,
        )
        # Report metrics with real sample counts, not expert-inflated weights.
        aggregation_fn = getattr(self, "fit_metrics_aggregation_fn", None)
        if aggregation_fn is not None:
            metrics_aggregated = aggregation_fn(
                [(fit_res.num_examples, fit_res.metrics or {}) for _, fit_res in results]
            )
        if self.tracker is not None:
            self.tracker.record_fit_round(server_round, metrics_aggregated)
        self._latest_params = parameters_aggregated
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round=server_round,
            results=results,
            failures=failures,
        )
        if self.tracker is not None:
            self.tracker.record_evaluate_round(
                server_round=server_round,
                distributed_loss=loss_aggregated,
                metrics=metrics_aggregated,
            )

        metric_value = metrics_aggregated.get(self.monitor_metric) if metrics_aggregated else None
        if metric_value is not None and float(metric_value) > self._best_metric:
            self._best_metric = float(metric_value)
            self._best_round = server_round
            self._best_params = self._latest_params

        return loss_aggregated, metrics_aggregated


class ReportingScaffold(ReportingFedAvg):
    """Reporting strategy with SCAFFOLD control-variate synchronization."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.c_global: list[np.ndarray] | None = None

    def configure_fit(self, server_round, parameters, client_manager):
        if self.c_global is None:
            ndarrays = parameters_to_ndarrays(parameters)
            self.c_global = [np.zeros_like(param, dtype=np.float32) for param in ndarrays]

        configured = super().configure_fit(server_round, parameters, client_manager)
        payload = pickle.dumps(self.c_global)
        updated = []
        for client_proxy, fit_ins in configured:
            config = dict(fit_ins.config)
            config["scaffold_c_global"] = payload
            updated.append(
                (
                    client_proxy,
                    FitIns(parameters=fit_ins.parameters, config=config),
                )
            )
        return updated

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        delta_c_list: list[list[np.ndarray]] = []
        for _, fit_res in results:
            raw_delta = (fit_res.metrics or {}).get("scaffold_delta_c")
            if raw_delta is None:
                raise ValueError(
                    "SCAFFOLD client result missing scaffold_delta_c. "
                    "All scaffold clients must return control-variate deltas."
                )
            delta_c_list.append(pickle.loads(raw_delta))

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
        )

        if self.c_global is not None and delta_c_list:
            for idx in range(len(self.c_global)):
                self.c_global[idx] = self.c_global[idx] + (
                    sum(delta[idx] for delta in delta_c_list) / len(delta_c_list)
                )

        return parameters_aggregated, metrics_aggregated
