from __future__ import annotations

import argparse
import logging

import flwr as fl

from src.common.config import load_yaml_config

logger = logging.getLogger("run_server")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Flower server for local V1")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--min-clients", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()

    cfg = {}
    if args.config is not None:
        cfg = load_yaml_config(args.config)

    server_cfg = cfg.get("server", {})

    host = args.host if args.host is not None else server_cfg.get("host", "0.0.0.0")
    port = args.port if args.port is not None else server_cfg.get("port", 8080)
    num_rounds = (
        args.num_rounds if args.num_rounds is not None else server_cfg.get("num_rounds", 3)
    )
    min_clients = (
        args.min_clients if args.min_clients is not None else server_cfg.get("min_clients", 3)
    )

    logger.info(
        "Starting Flower server | host=%s | port=%s | num_rounds=%s | min_clients=%s",
        host,
        port,
        num_rounds,
        min_clients,
    )

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )

    fl.server.start_server(
        server_address=f"{host}:{port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()