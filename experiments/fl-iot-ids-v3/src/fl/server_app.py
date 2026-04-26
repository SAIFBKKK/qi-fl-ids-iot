from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from src.common.logger import get_logger
from src.fl.aggregation_hooks import aggregate_evaluate_metrics, aggregate_fit_metrics
from src.fl.reporting_strategy import ReportingFedAvg, ReportingScaffold
from src.tracking.artifact_logger import BaselineArtifactTracker


logger = get_logger("fl_server")


def _default_config_from_run_config(run_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "project": {"seed": 42},
        "dataset": {"feature_count": 28, "num_classes": 34},
        "scenario": {"name": "normal_noniid", "num_clients": 3},
        "experiment": {"fl_strategy": str(run_config.get("strategy", "fedavg"))},
        "strategy": {
            "name": str(run_config.get("strategy", "fedavg")),
            "num_rounds": int(run_config.get("num-server-rounds", 3)),
            "fraction_train": 1.0,
            "fraction_evaluate": 1.0,
            "min_train_nodes": 3,
            "min_evaluate_nodes": 3,
            "min_available_nodes": 3,
            "expert_factor": 1.0,
            "expert_node_id": "node3",
        },
        "evaluation": {"best_round_monitor": "macro_f1"},
    }


def build_server_components(
    config: Mapping[str, Any],
    tracker: BaselineArtifactTracker | None = None,
    round_metric_logger: Callable[[int, dict[str, float]], None] | None = None,
) -> ServerAppComponents:
    strategy_name = str(
        config.get("experiment", {}).get(
            "fl_strategy",
            config.get("strategy", {}).get("name", "fedavg"),
        )
    ).lower()
    if strategy_name not in {"fedavg", "fedprox", "scaffold"}:
        raise ValueError(
            f"Unsupported FL strategy {strategy_name!r}. "
            "Supported strategies: fedavg, fedprox, scaffold."
        )

    strategy_cfg = dict(config.get("strategy", {}))
    num_rounds = int(strategy_cfg.get("num_rounds", 3))
    strategy_cls = ReportingScaffold if strategy_name == "scaffold" else ReportingFedAvg
    strategy = strategy_cls(
        tracker=tracker,
        monitor_metric=str(config.get("evaluation", {}).get("best_round_monitor", "macro_f1")),
        expert_node_id=str(strategy_cfg.get("expert_node_id", "node3")),
        expert_factor=float(strategy_cfg.get("expert_factor", 1.0)),
        fraction_fit=float(strategy_cfg.get("fraction_train", 1.0)),
        fraction_evaluate=float(strategy_cfg.get("fraction_evaluate", 1.0)),
        min_fit_clients=int(strategy_cfg.get("min_train_nodes", 3)),
        min_evaluate_clients=int(strategy_cfg.get("min_evaluate_nodes", 3)),
        min_available_clients=int(strategy_cfg.get("min_available_nodes", 3)),
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        round_metric_logger=round_metric_logger,
    )
    if tracker is not None:
        tracker.strategy = strategy

    logger.info(
        "ServerApp starting | strategy=%s | rounds=%s | clients=%s",
        strategy_name,
        num_rounds,
        int(config.get("scenario", {}).get("num_clients", 3)),
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))


def server_fn(context: Context) -> ServerAppComponents:
    return build_server_components(_default_config_from_run_config(context.run_config))


def create_server_app(
    config: Mapping[str, Any],
    tracker: BaselineArtifactTracker | None = None,
    round_metric_logger: Callable[[int, dict[str, float]], None] | None = None,
) -> ServerApp:
    def configured_server_fn(_: Context) -> ServerAppComponents:
        return build_server_components(
            config,
            tracker=tracker,
            round_metric_logger=round_metric_logger,
        )

    return ServerApp(server_fn=configured_server_fn)


app = ServerApp(server_fn=server_fn)
