from __future__ import annotations

from src.common.logger import get_logger
from src.fl.server.aggregation_hooks import (
    aggregate_evaluate_metrics,
    aggregate_fit_metrics,
)
from src.fl.server.reporting_strategy import ReportingFedAvg
from src.tracking.artifact_logger import BaselineArtifactTracker

logger = get_logger(__name__)


def build_strategy(
    strategy_name: str,
    config: dict,
    tracker: BaselineArtifactTracker | None = None,
):
    strategy_name = strategy_name.lower()

    num_rounds = int(config["strategy"]["num_rounds"])
    fraction_train = float(config["strategy"]["fraction_train"])
    fraction_evaluate = float(config["strategy"]["fraction_evaluate"])
    min_train_nodes = int(config["strategy"]["min_train_nodes"])
    min_evaluate_nodes = int(config["strategy"]["min_evaluate_nodes"])
    min_available_nodes = int(config["strategy"]["min_available_nodes"])
    monitor_metric = str(config.get("evaluation", {}).get("best_round_monitor", "macro_f1"))

    if strategy_name == "fedavg":
        logger.info("Building FedAvg strategy")
        return ReportingFedAvg(
            tracker=tracker,
            monitor_metric=monitor_metric,
            fraction_fit=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_train_nodes,
            min_evaluate_clients=min_evaluate_nodes,
            min_available_clients=min_available_nodes,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        ), num_rounds

    if strategy_name == "fedprox":
        logger.info("FedProx uses FedAvg server aggregation with client-side proximal loss")
        return ReportingFedAvg(
            tracker=tracker,
            monitor_metric=monitor_metric,
            fraction_fit=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_train_nodes,
            min_evaluate_clients=min_evaluate_nodes,
            min_available_clients=min_available_nodes,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        ), num_rounds

    if strategy_name == "scaffold":
        raise NotImplementedError(
            "SCAFFOLD is not implemented in fl-iot-ids-v2. "
            "Use fl-iot-ids-v3 for standard SCAFFOLD experiments."
        )

    raise ValueError(f"Unsupported strategy: {strategy_name}")
