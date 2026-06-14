from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from flwr.common import Context
try:
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
except ImportError:
    from flwr.server import ServerApp, ServerConfig

    @dataclass
    class ServerAppComponents:
        strategy: Any
        config: ServerConfig

from src.common.logger import get_logger
from src.common.paths import CONFIGS_DIR
from src.fl.aggregation_hooks import aggregate_evaluate_metrics, aggregate_fit_metrics
from src.fl.node_profiler import NodeProfiler
from src.fl.qifa_guard_strategy import ReportingQIFAGuard
from src.fl.qifa_strategy import ReportingQIFA
from src.fl.reporting_strategy import ReportingFedAvg, ReportingScaffold
from src.tracking.artifact_logger import BaselineArtifactTracker


logger = get_logger("fl_server")


def _server_app_supports_server_fn() -> bool:
    return "server_fn" in inspect.signature(ServerApp).parameters


def _resolve_tier_profiles_path(config: Mapping[str, Any]) -> Path:
    raw_path = dict(config.get("nodes", {})).get("tier_profiles_path")
    if raw_path is None:
        return CONFIGS_DIR / "nodes" / "tier_profiles.yaml"
    path = Path(str(raw_path))
    return path if path.is_absolute() else CONFIGS_DIR / path


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
    if strategy_name not in {"fedavg", "fedprox", "scaffold", "qifa", "qifa_guard"}:
        raise ValueError(
            f"Unsupported FL strategy {strategy_name!r}. "
            "Supported strategies: fedavg, fedprox, scaffold, qifa, qifa_guard."
        )

    strategy_cfg = dict(config.get("strategy", {}))
    num_rounds = int(strategy_cfg.get("num_rounds", 3))
    if strategy_name == "scaffold":
        strategy_cls = ReportingScaffold
    elif strategy_name == "qifa":
        strategy_cls = ReportingQIFA
    elif strategy_name == "qifa_guard":
        strategy_cls = ReportingQIFAGuard
    else:
        strategy_cls = ReportingFedAvg
    model_cfg = dict(config.get("model", {}))
    node_profiler = NodeProfiler(_resolve_tier_profiles_path(config))
    model_config = {
        "input_dim": int(
            model_cfg.get("input_dim", config.get("dataset", {}).get("feature_count", 28))
        ),
        "num_classes": int(
            model_cfg.get("output_dim", config.get("dataset", {}).get("num_classes", 34))
        ),
        "hidden_dims": model_cfg.get(
            "hidden_dims",
            [
                model_cfg.get("max_hidden_1", 256),
                model_cfg.get("max_hidden_2", 128),
            ],
        ),
        "max_hidden_1": int(model_cfg.get("max_hidden_1", 256)),
        "max_hidden_2": int(model_cfg.get("max_hidden_2", 128)),
        "dropout": float(model_cfg.get("dropout", 0.2)),
    }
    strategy_kwargs: dict[str, Any] = {
        "tracker": tracker,
        "monitor_metric": str(config.get("evaluation", {}).get("best_round_monitor", "macro_f1")),
        "experiment_id": str(config.get("experiment", {}).get("name", "unknown")),
        "scenario": str(config.get("scenario", {}).get("name", "normal_noniid")),
        "common_global_validation": bool(config.get("evaluation", {}).get("common_global_validation", False)),
        "expert_node_id": str(strategy_cfg.get("expert_node_id", "node3")),
        "expert_factor": float(strategy_cfg.get("expert_factor", 1.0)),
        "fraction_fit": float(strategy_cfg.get("fraction_train", 1.0)),
        "fraction_evaluate": float(strategy_cfg.get("fraction_evaluate", 1.0)),
        "min_fit_clients": int(strategy_cfg.get("min_train_nodes", 3)),
        "min_evaluate_clients": int(strategy_cfg.get("min_evaluate_nodes", 3)),
        "min_available_clients": int(strategy_cfg.get("min_available_nodes", 3)),
        "fit_metrics_aggregation_fn": aggregate_fit_metrics,
        "evaluate_metrics_aggregation_fn": aggregate_evaluate_metrics,
        "round_metric_logger": round_metric_logger,
        "output_dir": tracker.report_dir if tracker is not None else None,
        "node_profiler": node_profiler,
        "model_config": model_config,
        "supernet_config": model_config,
        "multitier_enabled": bool(strategy_cfg.get("multitier_enabled", False)),
    }
    if strategy_name in {"qifa", "qifa_guard"}:
        qifa_cfg = dict(strategy_cfg.get("qifa", {}))
        if strategy_name == "qifa_guard":
            strategy_kwargs.update(
                {
                    "lambda_qifa": float(qifa_cfg.get("lambda_qifa", 0.0)),
                    "beta_loss": float(qifa_cfg.get("beta_loss", 0.0)),
                    "rho_rare": float(qifa_cfg.get("rho_rare", 0.0)),
                    "min_client_weight": qifa_cfg.get("min_client_weight"),
                    "max_client_weight": qifa_cfg.get("max_client_weight"),
                    "use_global_val_quality": bool(qifa_cfg.get("use_global_val_quality", True)),
                    "use_rare_bonus": bool(qifa_cfg.get("use_rare_bonus", True)),
                    "perturbation_enabled": bool(qifa_cfg.get("perturbation_enabled", False)),
                    "random_seed": int(qifa_cfg.get("random_seed", config.get("project", {}).get("seed", 42))),
                }
            )
        else:
            strategy_kwargs.update(
                {
                    "lambda_qifa": float(qifa_cfg.get("lambda_qifa", 0.0)),
                    "perturbation_enabled": bool(qifa_cfg.get("perturbation_enabled", False)),
                    "delta_perturbation": float(qifa_cfg.get("delta_perturbation", 0.0)),
                    "sigma_noise": float(qifa_cfg.get("sigma_noise", 0.0)),
                    "perturbation_frequency": int(qifa_cfg.get("perturbation_frequency", 1)),
                    "random_seed": int(qifa_cfg.get("random_seed", config.get("project", {}).get("seed", 42))),
                }
            )
    strategy = strategy_cls(**strategy_kwargs)
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
    if not _server_app_supports_server_fn():
        components = build_server_components(
            config,
            tracker=tracker,
            round_metric_logger=round_metric_logger,
        )
        return ServerApp(config=components.config, strategy=components.strategy)

    def configured_server_fn(_: Context) -> ServerAppComponents:
        return build_server_components(
            config,
            tracker=tracker,
            round_metric_logger=round_metric_logger,
        )

    return ServerApp(server_fn=configured_server_fn)


if _server_app_supports_server_fn():
    app = ServerApp(server_fn=server_fn)
else:
    app = ServerApp()
