"""Mock Flower FL client for the P5 training Docker profile."""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
from loguru import logger


NDArrays = List[np.ndarray]
ScalarMetrics = Dict[str, float | int | str | bool]


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=os.getenv("LOG_LEVEL", "INFO"),
        serialize=os.getenv("LOG_FORMAT", "json").lower() == "json",
        backtrace=False,
        diagnose=False,
    )


def read_int_env(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError:
        logger.critical("{} must be an integer, got {!r}", name, raw_value)
        sys.exit(1)
    if value < minimum:
        logger.critical("{} must be >= {}, got {}", name, minimum, value)
        sys.exit(1)
    return value


def read_float_env(name: str, default: float, minimum: float = 0.0) -> float:
    raw_value = os.getenv(name, str(default))
    try:
        value = float(raw_value)
    except ValueError:
        logger.critical("{} must be a float, got {!r}", name, raw_value)
        sys.exit(1)
    if value < minimum:
        logger.critical("{} must be >= {}, got {}", name, minimum, value)
        sys.exit(1)
    return value


class MockFlowerClient(fl.client.NumPyClient):
    """Small deterministic client used only to validate training orchestration."""

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.num_examples = read_int_env("MOCK_NUM_EXAMPLES", 128)
        client_seed = sum(ord(char) for char in client_id)
        rng = np.random.default_rng(client_seed)
        self.parameters: NDArrays = [
            rng.normal(0.0, 0.01, size=(4,)).astype(np.float32),
            rng.normal(0.0, 0.01, size=(2, 2)).astype(np.float32),
        ]

    def get_parameters(self, config: Dict[str, str]) -> NDArrays:
        logger.info("Client {} get_parameters config={}", self.client_id, config)
        return self.parameters

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, str],
    ) -> Tuple[NDArrays, int, ScalarMetrics]:
        server_round = int(config.get("server_round", 0) or 0)
        logger.info("Client {} fit round={}", self.client_id, server_round)
        self.parameters = [array + np.float32(0.001) for array in parameters]
        return self.parameters, self.num_examples, {
            "client_id": self.client_id,
            "server_round": server_round,
            "mock_fit": 1,
        }

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, str],
    ) -> Tuple[float, int, ScalarMetrics]:
        server_round = int(config.get("server_round", 0) or 0)
        loss = float(sum(np.mean(np.abs(array)) for array in parameters))
        logger.info(
            "Client {} evaluate round={} loss={:.6f}",
            self.client_id,
            server_round,
            loss,
        )
        return loss, self.num_examples, {
            "client_id": self.client_id,
            "server_round": server_round,
            "mock_evaluate": 1,
        }


def connect_with_retry(server_address: str, client: MockFlowerClient) -> None:
    retries = read_int_env("CLIENT_CONNECT_RETRIES", 20)
    base_delay = read_float_env("CLIENT_RETRY_DELAY_SECONDS", 2.0)

    for attempt in range(1, retries + 1):
        try:
            logger.info(
                "Client {} connecting to Flower server {} (attempt {}/{})",
                client.client_id,
                server_address,
                attempt,
                retries,
            )
            fl.client.start_numpy_client(server_address=server_address, client=client)
            logger.info("Client {} finished cleanly", client.client_id)
            return
        except Exception as exc:  # pragma: no cover - depends on runtime network
            if attempt == retries:
                logger.critical(
                    "Client {} could not connect to {} after {} attempts: {}",
                    client.client_id,
                    server_address,
                    retries,
                    exc,
                )
                sys.exit(1)
            delay = min(base_delay * attempt, 15.0)
            logger.warning(
                "Client {} connection failed: {}; retrying in {:.1f}s",
                client.client_id,
                exc,
                delay,
            )
            time.sleep(delay)


def main() -> None:
    configure_logging()

    training_mode = os.getenv("TRAINING_MODE", "mock")
    if training_mode != "mock":
        logger.critical("P5 only supports TRAINING_MODE=mock, got {}", training_mode)
        sys.exit(1)

    client_id = os.getenv("CLIENT_ID", "client1")
    server_address = os.getenv("FL_SERVER_ADDRESS", "fl-server:8080")
    logger.info(
        "Starting mock Flower client {} for orchestration-only training",
        client_id,
    )

    connect_with_retry(
        server_address=server_address,
        client=MockFlowerClient(client_id=client_id),
    )


if __name__ == "__main__":
    main()
