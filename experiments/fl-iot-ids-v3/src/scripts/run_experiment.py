from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from flwr.simulation import run_simulation

from src.common.config import load_experiment_bundle
from src.common.logger import get_logger
from src.common.paths import CONFIGS_DIR, ensure_runtime_dirs
from src.common.registry import find_experiment
from src.common.utils import get_expected_node_ids, set_seed
from src.fl.client_app import create_client_app
from src.fl.server_app import create_server_app
from src.tracking.artifact_logger import BaselineArtifactTracker
from src.tracking.run_naming import (
    generate_experiment_display_name,
    generate_run_name,
)
from src.utils.mlflow_logger import MLflowRunLogger

logger = get_logger("run_experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an fl-iot-ids-v3 registered experiment")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name from configs/experiment_registry.yaml",
    )
    return parser.parse_args()


def load_experiment_config(experiment_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    experiment = find_experiment(experiment_name)
    fl_cfg_name = str(experiment.get("fl_config", experiment["fl_strategy"]))

    config = load_experiment_bundle(
        global_cfg_path=CONFIGS_DIR / "global.yaml",
        fl_cfg_path=CONFIGS_DIR / "fl" / f"{fl_cfg_name}.yaml",
        model_cfg_path=CONFIGS_DIR / "model" / f"{experiment['architecture']}.yaml",
        data_cfg_path=CONFIGS_DIR / "data" / f"{experiment['data_scenario']}.yaml",
        imbalance_cfg_path=CONFIGS_DIR / "imbalance" / f"{experiment['imbalance_strategy']}.yaml",
    )

    config["experiment"] = dict(experiment)
    config.setdefault("project", {})
    config["project"].setdefault("name", "fl-iot-ids-v3")
    config["project"].setdefault("seed", int(config.get("seed", 42)))

    config.setdefault("strategy", {})
    config["strategy"]["name"] = str(experiment["fl_strategy"])
    if "expert_factor" in experiment:
        config["strategy"]["expert_factor"] = float(experiment["expert_factor"])

    config.setdefault("scenario", {})
    config["scenario"]["name"] = str(experiment["data_scenario"])
    config.setdefault("imbalance", {})
    config["imbalance"]["name"] = str(experiment["imbalance_strategy"])

    config.setdefault("mlflow", {})
    config["mlflow"].setdefault("enabled", True)
    config["mlflow"].setdefault("tracking_uri", "./outputs/mlruns")

    return experiment, config


def flatten_params(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_params(dotted, value))
        elif isinstance(value, (str, int, float, bool)):
            flattened[dotted] = value
    return flattened


def save_resolved_config(report_dir: Path, experiment: dict[str, Any], config: dict[str, Any]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = report_dir / "resolved_config.json"
    payload = {
        "experiment": experiment,
        "config": config,
    }
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return resolved_config_path


def log_experiment_to_mlflow(
    mlflow_logger: MLflowRunLogger,
    experiment: dict[str, Any],
    config: dict[str, Any],
) -> None:
    params = {}
    params.update(flatten_params("experiment", experiment))
    params.update(flatten_params("config", config))
    mlflow_logger.log_params(params)


def log_artifacts_to_mlflow(
    mlflow_logger: MLflowRunLogger,
    report_dir: Path,
    summary: dict[str, Any],
) -> None:
    metric_payload = {
        "duration_sec": float(summary.get("duration_sec", 0.0)),
        "completed_rounds": float(summary.get("completed_rounds", 0)),
    }
    for key, value in summary.items():
        if key.startswith("final_") and isinstance(value, (int, float)):
            metric_payload[key] = float(value)
    mlflow_logger.log_metrics(metric_payload)

    for artifact_name in (
        "resolved_config.json",
        "run_summary.json",
        "round_metrics.json",
        "baseline_notes.md",
    ):
        artifact_path = report_dir / artifact_name
        if artifact_path.exists():
            mlflow_logger.log_artifact(artifact_path, artifact_path="reports")


def main() -> None:
    args = parse_args()
    experiment, config = load_experiment_config(args.experiment)

    ensure_runtime_dirs()
    set_seed(int(config["project"].get("seed", 42)))

    tracker = BaselineArtifactTracker(experiment=experiment, config=config)
    report_dir = tracker.report_dir
    resolved_config_path = save_resolved_config(report_dir, experiment, config)

    mlflow_cfg = dict(config.get("mlflow", {}))
    mlflow_enabled = bool(mlflow_cfg.get("enabled", True))
    mlflow_logger: MLflowRunLogger | None = None
    if mlflow_enabled:
        mlflow_logger = MLflowRunLogger(
            tracking_uri=str(mlflow_cfg.get("tracking_uri", "./outputs/mlruns")),
            experiment_name=generate_experiment_display_name(experiment),
            run_name=generate_run_name(experiment),
        )
        mlflow_logger.start()
        log_experiment_to_mlflow(mlflow_logger, experiment, config)
        mlflow_logger.log_artifact(resolved_config_path, artifact_path="reports")

    server_app = create_server_app(config, tracker=tracker)
    client_app = create_client_app(config)

    num_clients = int(config["scenario"]["num_clients"])
    node_ids = get_expected_node_ids(num_clients)
    config["scenario"]["node_ids"] = node_ids
    logger.info("Selected clients: %s", ", ".join(node_ids))

    backend_config: dict[str, Any] = {
        "init_args": {"include_dashboard": False},
        "client_resources": {"num_cpus": 1},
    }
    if os.name == "nt":
        backend_config["init_args"]["num_cpus"] = 1

    start = perf_counter()
    status = "success"
    error_message: str | None = None

    try:
        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=num_clients,
            backend_name="ray",
            backend_config=backend_config,
        )
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        raise
    finally:
        duration_sec = perf_counter() - start
        tracker.save_baseline_artifacts(
            status=status,
            duration_sec=duration_sec,
            error_message=error_message,
        )

        if mlflow_logger is not None:
            summary_path = report_dir / "run_summary.json"
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
                log_artifacts_to_mlflow(mlflow_logger, report_dir, summary)
            mlflow_logger.end()


if __name__ == "__main__":
    main()
