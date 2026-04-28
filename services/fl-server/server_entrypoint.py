"""Flower training dispatcher for the Compose training profile.

TRAINING_MODE=mock keeps the P5 lightweight Flower orchestration.
TRAINING_MODE=real delegates to the validated scientific runner mounted from
experiments/fl-iot-ids-v3 without modifying that source tree.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import nullcontext
from typing import Any

import flwr as fl
import mlflow
from flwr.server.strategy import FedAvg
from loguru import logger


DEFAULT_NUM_ROUNDS = 10
DEFAULT_REAL_EXPERIMENT = "exp_v4_multitier_fedavg_normal_classweights"
DEFAULT_REAL_WORKDIR = "/app/experiments/fl-iot-ids-v3"


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


def start_mlflow_run(training_mode: str, num_rounds: int) -> Any:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.warning("MLFLOW_TRACKING_URI is not set; MLflow logging disabled")
        return nullcontext()

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "p5_mock_fl_training"))
        run = mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME", "p5-mock-fl-server"))
        mlflow.log_param("training_mode", training_mode)
        mlflow.log_param("num_rounds", num_rounds)
        mlflow.log_param("strategy", "FedAvg")
        mlflow.log_metric("server_started", 1.0)
        return run
    except Exception as exc:  # pragma: no cover - depends on runtime MLflow
        logger.warning("MLflow logging disabled after startup failure: {}", exc)
        return nullcontext()


def log_mlflow_metric(name: str, value: float) -> None:
    if mlflow.active_run() is None:
        return
    try:
        mlflow.log_metric(name, value)
    except Exception as exc:  # pragma: no cover - depends on runtime MLflow
        logger.warning("Could not log {} to MLflow: {}", name, exc)


def keep_alive_after_training() -> None:
    keep_alive = os.getenv("KEEP_SERVER_ALIVE", "true").lower() in {"1", "true", "yes"}
    if not keep_alive:
        return

    logger.info("Mock FL training finished; keeping fl-server container alive")
    while True:
        time.sleep(60)


def run_mock_training(training_mode: str) -> None:
    host = os.getenv("FL_SERVER_HOST", "0.0.0.0")
    port = read_int_env("FL_SERVER_PORT", 8080)
    num_rounds = read_int_env("FL_NUM_ROUNDS", DEFAULT_NUM_ROUNDS)

    server_address = f"{host}:{port}"
    logger.info(
        "Starting mock Flower training server on {} for {} rounds",
        server_address,
        num_rounds,
    )
    logger.info(
        "P5 mock training profile validates orchestration, not scientific FL metrics"
    )

    strategy = FedAvg(
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )

    with start_mlflow_run(training_mode=training_mode, num_rounds=num_rounds):
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
        logger.info("Mock Flower training finished after {} rounds", num_rounds)
        log_mlflow_metric("training_finished", 1.0)

    keep_alive_after_training()


def run_real_training() -> None:
    experiment = os.getenv("REAL_FL_EXPERIMENT", DEFAULT_REAL_EXPERIMENT)
    rounds = read_int_env("REAL_FL_ROUNDS", 1)
    workdir = Path(os.getenv("REAL_FL_WORKDIR", DEFAULT_REAL_WORKDIR))
    tracking_uri = "http://mlflow:5000"
    artifact_root = "/app/experiments/fl-iot-ids-v3/outputs/mlruns"
    git_python_refresh = "quiet"

    if not workdir.exists():
        logger.critical(
            "Real FL workdir is missing: {}. Mount experiments/fl-iot-ids-v3 first.",
            workdir,
        )
        sys.exit(1)

    runner = workdir / "src" / "scripts" / "run_experiment.py"
    if not runner.exists():
        logger.critical("Real FL runner not found: {}", runner)
        sys.exit(1)

    command = [
        sys.executable,
        "/app/real_runner_wrapper.py",
        "--experiment",
        experiment,
        "--rounds",
        str(rounds),
    ]

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = tracking_uri
    env["MLFLOW_ARTIFACT_ROOT"] = artifact_root
    env["GIT_PYTHON_REFRESH"] = git_python_refresh
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(workdir)
        if not existing_pythonpath
        else f"{workdir}{os.pathsep}{existing_pythonpath}"
    )

    logger.info(
        "TRAINING_MODE=real: launching scientific runner experiment={} rounds={} workdir={}",
        experiment,
        rounds,
        workdir,
    )
    logger.info(
        "P6A-lite real mode uses simulation-based run_experiment.py, not multi-container clients"
    )
    logger.info("MLFLOW_TRACKING_URI injected as {}", tracking_uri)
    logger.info("MLFLOW_ARTIFACT_ROOT injected as {}", artifact_root)
    logger.info("GIT_PYTHON_REFRESH injected as {}", git_python_refresh)

    result = subprocess.run(command, cwd=str(workdir), env=env, check=False)
    if result.returncode != 0:
        logger.critical(
            "Scientific runner failed with exit code {}", result.returncode
        )
        sys.exit(result.returncode)

    logger.info("Scientific runner completed successfully")


def main() -> None:
    configure_logging()

    training_mode = os.getenv("TRAINING_MODE", "mock").lower()

    if training_mode == "mock":
        run_mock_training(training_mode=training_mode)
        return
    if training_mode == "real":
        run_real_training()
        return

    logger.critical("Unsupported TRAINING_MODE={!r}; expected mock or real", training_mode)
    sys.exit(1)


if __name__ == "__main__":
    main()
