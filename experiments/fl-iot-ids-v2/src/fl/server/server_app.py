from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from flwr.app import Context
from flwr.server import ServerAppComponents, ServerConfig
from flwr.serverapp import ServerApp

from src.common.logger import get_logger
from src.fl.server.strategy_factory import build_strategy
from src.tracking.artifact_logger import BaselineArtifactTracker

logger = get_logger(__name__)


def _read_config_value(
    config: Mapping[str, Any],
    nested_path: tuple[str, ...],
    *flat_keys: str,
) -> Any:
    current: Any = config
    for key in nested_path:
        if isinstance(current, Mapping) and key in current:
            current = current[key]
        else:
            break
    else:
        return current

    for key in flat_keys:
        if key in config:
            return config[key]

    raise KeyError(
        f"Missing configuration for {'.'.join(nested_path)}. "
        f"Tried flat keys: {', '.join(flat_keys)}"
    )


def build_server_components(
    config: Mapping[str, Any],
    tracker: BaselineArtifactTracker | None = None,
) -> ServerAppComponents:
    try:
        strategy_name = str(
            _read_config_value(
                config,
                ("experiment", "fl_strategy"),
                "experiment.fl_strategy",
                "fl_strategy",
            )
        )
    except KeyError:
        strategy_name = str(
            _read_config_value(
                config,
                ("strategy", "name"),
                "strategy.name",
            )
        )
    num_rounds = int(
        _read_config_value(
            config,
            ("strategy", "num_rounds"),
            "strategy.num_rounds",
            "num_rounds",
        )
    )

    strategy_config = {
        "strategy": {
            "num_rounds": num_rounds,
            "fraction_train": float(
                _read_config_value(
                    config,
                    ("strategy", "fraction_train"),
                    "strategy.fraction_train",
                    "fraction_train",
                )
            ),
            "fraction_evaluate": float(
                _read_config_value(
                    config,
                    ("strategy", "fraction_evaluate"),
                    "strategy.fraction_evaluate",
                    "fraction_evaluate",
                )
            ),
            "min_train_nodes": int(
                _read_config_value(
                    config,
                    ("strategy", "min_train_nodes"),
                    "strategy.min_train_nodes",
                    "min_train_nodes",
                )
            ),
            "min_evaluate_nodes": int(
                _read_config_value(
                    config,
                    ("strategy", "min_evaluate_nodes"),
                    "strategy.min_evaluate_nodes",
                    "min_evaluate_nodes",
                )
            ),
            "min_available_nodes": int(
                _read_config_value(
                    config,
                    ("strategy", "min_available_nodes"),
                    "strategy.min_available_nodes",
                    "min_available_nodes",
                )
            ),
        }
    }

    strategy, resolved_rounds = build_strategy(
        strategy_name,
        strategy_config,
        tracker=tracker,
    )

    logger.info(
        "ServerApp initialized with strategy=%s, rounds=%s",
        strategy_name,
        resolved_rounds,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=resolved_rounds),
    )


def server_fn(context: Context) -> ServerAppComponents:
    return build_server_components(context.run_config)


def create_server_app(
    config: Mapping[str, Any],
    tracker: BaselineArtifactTracker | None = None,
) -> ServerApp:
    def configured_server_fn(_: Context) -> ServerAppComponents:
        return build_server_components(config, tracker=tracker)

    return ServerApp(server_fn=configured_server_fn)


app = ServerApp(server_fn=server_fn)
