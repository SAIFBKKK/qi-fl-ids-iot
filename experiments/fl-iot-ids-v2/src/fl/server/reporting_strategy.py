from __future__ import annotations

import math

from flwr.common import EvaluateRes, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.tracking.artifact_logger import BaselineArtifactTracker


class ReportingFedAvg(FedAvg):
    def __init__(
        self,
        *,
        tracker: BaselineArtifactTracker | None = None,
        monitor_metric: str = "macro_f1",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tracker = tracker
        self.monitor_metric = monitor_metric
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

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
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
