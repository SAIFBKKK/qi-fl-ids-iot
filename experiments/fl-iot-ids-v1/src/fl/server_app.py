from __future__ import annotations

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from src.common.logger import get_logger
from src.fl.metrics import weighted_average


logger = get_logger("fl_server")


def server_fn(context: Context) -> ServerAppComponents:
    run_config = context.run_config
    num_rounds = int(run_config["num-server-rounds"])

    logger.info("ServerApp starting | rounds=%s", num_rounds)

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)