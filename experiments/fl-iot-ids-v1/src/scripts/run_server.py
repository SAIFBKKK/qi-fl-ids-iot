from __future__ import annotations

import time
from pathlib import Path
import argparse
import logging

import flwr as fl

from src.utils.mlflow_logger import MLflowRunLogger
from src.common.config import load_yaml_config

logger = logging.getLogger("run_server")


def weighted_average(metrics):
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated = {}

    for num_examples, metric_dict in metrics:
        for key, value in metric_dict.items():
            aggregated[key] = aggregated.get(key, 0.0) + float(value) * num_examples

    for key in aggregated:
        aggregated[key] /= total_examples

    return aggregated


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

    mlflow_cfg = cfg.get("mlflow", {})
    mlflow_enabled = bool(mlflow_cfg.get("enabled", False))
    mlflow_logger = None

    if mlflow_enabled:
        mlflow_logger = MLflowRunLogger(
            tracking_uri=mlflow_cfg.get("tracking_uri", "file:/app/outputs/mlruns"),
            experiment_name=mlflow_cfg.get("experiment_name", "fl-iot-ids-v1"),
            run_name=mlflow_cfg.get("run_name", "baseline-v1"),
        )
        mlflow_logger.start()

    server_cfg = cfg.get("server", {})
    client_cfg = cfg.get("client", {})

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

    if mlflow_logger is not None:
        mlflow_logger.log_params(
            {
                "model": "MLP",
                "strategy": "FedAvg",
                "host": host,
                "port": port,
                "num_rounds": num_rounds,
                "min_clients": min_clients,
                "local_epochs": client_cfg.get("local_epochs", 1),
                "batch_size": client_cfg.get("batch_size", 256),
                "learning_rate": client_cfg.get("learning_rate", 0.001),
                "dataset": "CICIoT2023",
            }
        )

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    start_time = time.time()

    history = fl.server.start_server(
        server_address=f"{host}:{port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    elapsed = time.time() - start_time
    logger.info("Training finished | total_time_sec=%.2f", elapsed)

    if mlflow_logger is not None:
        mlflow_logger.log_metrics({"total_time_sec": elapsed})

        for rnd, loss in getattr(history, "losses_distributed", []):
            mlflow_logger.log_metrics({"global_loss": float(loss)}, step=rnd)

        for rnd, loss in getattr(history, "losses_centralized", []):
            mlflow_logger.log_metrics({"centralized_loss": float(loss)}, step=rnd)

        metrics_dist = getattr(history, "metrics_distributed", {})
        for metric_name, values in metrics_dist.items():
            for rnd, val in values:
                mlflow_logger.log_metrics(
                    {f"global_{metric_name}": float(val)},
                    step=rnd,
                )

        if args.config is not None:
            config_path = Path(__file__).resolve().parents[2] / args.config
            if config_path.exists():
                mlflow_logger.log_artifact(config_path, artifact_path="configs")

        mlflow_logger.end()


if __name__ == "__main__":
    main()