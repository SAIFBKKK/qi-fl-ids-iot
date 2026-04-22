from __future__ import annotations

from flwr.server.strategy import FedAvg

from src.fl.metrics import weighted_average


def build_strategy(
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 1,
    min_evaluate_clients: int = 1,
    min_available_clients: int = 1,
):
    return FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )