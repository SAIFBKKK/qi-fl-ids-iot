from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from flwr.clientapp import ClientApp
from flwr.simulation import run_simulation

from src.common.config import load_experiment_bundle
from src.common.paths import CONFIGS_DIR, OUTPUTS_DIR, ensure_runtime_dirs
from src.common.registry import find_experiment
from src.common.seeds import set_global_seed
from src.fl.server.server_app import create_server_app
from src.fl.simulation.client_factory import make_client_fn
from src.tracking.artifact_logger import BaselineArtifactTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FL IoT IDS v2 experiment")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name from configs/experiment_registry.yaml",
    )
    return parser.parse_args()


def load_experiment_config(experiment_name: str) -> tuple[dict, dict]:
    experiment = find_experiment(experiment_name)

    fl_cfg_name = experiment.get("fl_config", experiment["fl_strategy"])

    config = load_experiment_bundle(
        global_cfg_path=CONFIGS_DIR / "global.yaml",
        fl_cfg_path=CONFIGS_DIR / "fl" / f"{fl_cfg_name}.yaml",
        model_cfg_path=CONFIGS_DIR / "model" / f"{experiment['architecture']}.yaml",
        data_cfg_path=CONFIGS_DIR / "data" / f"{experiment['data_scenario']}.yaml",
        imbalance_cfg_path=CONFIGS_DIR / "imbalance" / f"{experiment['imbalance_strategy']}.yaml",
    )

    return experiment, config


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def history_to_dict(history: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "losses_distributed": [],
        "losses_centralized": [],
        "metrics_distributed_fit": {},
        "metrics_distributed": {},
        "metrics_centralized": {},
    }

    if hasattr(history, "losses_distributed"):
        payload["losses_distributed"] = [
            {"round": int(server_round), "loss": float(loss)}
            for server_round, loss in history.losses_distributed
        ]

    if hasattr(history, "losses_centralized"):
        payload["losses_centralized"] = [
            {"round": int(server_round), "loss": float(loss)}
            for server_round, loss in history.losses_centralized
        ]

    def convert_metric_map(metric_map: dict[str, list[tuple[int, Any]]]) -> dict[str, list[dict[str, Any]]]:
        converted: dict[str, list[dict[str, Any]]] = {}
        for key, values in metric_map.items():
            converted[key] = []
            for server_round, value in values:
                try:
                    cast_value: Any = float(value)
                except Exception:
                    cast_value = value
                converted[key].append(
                    {"round": int(server_round), "value": cast_value}
                )
        return converted

    if hasattr(history, "metrics_distributed_fit"):
        payload["metrics_distributed_fit"] = convert_metric_map(
            history.metrics_distributed_fit
        )

    if hasattr(history, "metrics_distributed"):
        payload["metrics_distributed"] = convert_metric_map(history.metrics_distributed)

    if hasattr(history, "metrics_centralized"):
        payload["metrics_centralized"] = convert_metric_map(history.metrics_centralized)

    return payload


def build_run_summary(
    experiment: dict,
    config: dict,
    history_dict: dict[str, Any],
    *,
    duration_sec: float,
    status: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    final_loss = None
    if history_dict["losses_distributed"]:
        final_loss = history_dict["losses_distributed"][-1]["loss"]

    requested_rounds = int(config["strategy"]["num_rounds"])
    completed_rounds = len(history_dict["losses_distributed"])
    effective_status = status
    if status == "success" and completed_rounds < requested_rounds:
        effective_status = "partial"

    summary = {
        "generated_at": _utc_now_iso(),
        "experiment_name": experiment["name"],
        "architecture": experiment["architecture"],
        "fl_strategy": experiment["fl_strategy"],
        "data_scenario": experiment["data_scenario"],
        "imbalance_strategy": experiment["imbalance_strategy"],
        "num_rounds": requested_rounds,
        "completed_rounds": completed_rounds,
        "num_clients": int(config["scenario"]["num_clients"]),
        "feature_count": int(config["dataset"]["feature_count"]),
        "num_classes": int(config["dataset"]["num_classes"]),
        "duration_sec": round(float(duration_sec), 2),
        "final_distributed_loss": final_loss,
        "status": effective_status,
    }

    if error_message:
        summary["error_message"] = error_message

    for metric_name in [
        "accuracy",
        "macro_f1",
        "recall_macro",
        "benign_recall",
        "false_positive_rate",
        "rare_class_recall",
    ]:
        metric_series = history_dict["metrics_distributed"].get(metric_name, [])
        if metric_series:
            summary[f"final_{metric_name}"] = metric_series[-1]["value"]

    return summary


def save_baseline_artifacts(
    experiment: dict,
    config: dict,
    history: Any,
    *,
    duration_sec: float,
    status: str,
    error_message: str | None = None,
) -> None:
    report_dir = OUTPUTS_DIR / "reports" / "baselines" / experiment["name"]
    report_dir.mkdir(parents=True, exist_ok=True)

    history_dict = history_to_dict(history)
    summary = build_run_summary(
        experiment,
        config,
        history_dict,
        duration_sec=duration_sec,
        status=status,
        error_message=error_message,
    )

    with (report_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    round_rows: list[dict[str, Any]] = []
    rounds = sorted(
        {
            item["round"] for item in history_dict["losses_distributed"]
        }
        | {
            item["round"]
            for series in history_dict["metrics_distributed_fit"].values()
            for item in series
        }
        | {
            item["round"]
            for series in history_dict["metrics_distributed"].values()
            for item in series
        }
    )

    fit_metric_keys = ("train_loss_last", "train_time_sec", "update_size_bytes")
    eval_metric_keys = (
        "accuracy",
        "macro_f1",
        "recall_macro",
        "benign_recall",
        "false_positive_rate",
        "rare_class_recall",
    )

    loss_by_round = {
        item["round"]: item["loss"] for item in history_dict["losses_distributed"]
    }
    fit_by_metric = {
        key: {item["round"]: item["value"] for item in series}
        for key, series in history_dict["metrics_distributed_fit"].items()
    }
    eval_by_metric = {
        key: {item["round"]: item["value"] for item in series}
        for key, series in history_dict["metrics_distributed"].items()
    }

    for server_round in rounds:
        row: dict[str, Any] = {
            "round": int(server_round),
            "distributed_loss": loss_by_round.get(server_round),
        }
        for key in fit_metric_keys:
            row[key] = fit_by_metric.get(key, {}).get(server_round)
        for key in eval_metric_keys:
            row[key] = eval_by_metric.get(key, {}).get(server_round)
        round_rows.append(row)

    round_metrics_payload = {
        "generated_at": _utc_now_iso(),
        "experiment_name": experiment["name"],
        "rounds": round_rows,
        "history": history_dict,
    }

    with (report_dir / "round_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(round_metrics_payload, handle, indent=2)

    notes_lines = [
        "# Baseline officielle",
        "",
        f"- Experiment: {experiment['name']}",
        f"- Architecture: {experiment['architecture']}",
        f"- Strategy: {experiment['fl_strategy']}",
        f"- Scenario: {experiment['data_scenario']}",
        f"- Imbalance: {experiment['imbalance_strategy']}",
        f"- Num rounds: {config['strategy']['num_rounds']}",
        f"- Completed rounds: {summary['completed_rounds']}",
        f"- Num clients: {config['scenario']['num_clients']}",
        f"- Feature count: {config['dataset']['feature_count']}",
        f"- Num classes: {config['dataset']['num_classes']}",
        f"- Duration sec: {summary['duration_sec']}",
        f"- Status: {summary['status']}",
        "",
        "## Fichiers",
        "- run_summary.json",
        "- round_metrics.json",
        "- baseline_notes.md",
    ]

    if error_message:
        notes_lines.extend(["", "## Erreur", f"- {error_message}"])

    if summary.get("final_distributed_loss") is not None:
        notes_lines.extend(
            [
                "",
                "## Final metrics",
                f"- final_distributed_loss: {summary['final_distributed_loss']}",
            ]
        )
        for metric_name in eval_metric_keys:
            key = f"final_{metric_name}"
            if key in summary:
                notes_lines.append(f"- {key}: {summary[key]}")

    with (report_dir / "baseline_notes.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(notes_lines) + "\n")


def main() -> None:
    args = parse_args()

    experiment, config = load_experiment_config(args.experiment)

    ensure_runtime_dirs()
    set_global_seed(int(config["project"].get("seed", 42)))

    tracker = BaselineArtifactTracker(experiment=experiment, config=config)

    server_app = create_server_app(config, tracker=tracker)
    client_app = ClientApp(client_fn=make_client_fn(config))

    num_clients = int(config["scenario"]["num_clients"])
    backend_config = {
        "init_args": {
            "include_dashboard": False,
        },
        "client_resources": {"num_cpus": 1},
    }
    if os.name == "nt":
        backend_config["init_args"]["num_cpus"] = 1

    start = perf_counter()
    status = "success"
    error_message: str | None = None
    history: Any = None

    try:
        history = run_simulation(
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
        # Flower 1.29 returns None here, so we reconstruct History from the
        # tracker when the Python API does not expose it directly.
        if history is None:
            history = tracker.to_history()

        save_baseline_artifacts(
            experiment,
            config,
            history,
            status=status,
            duration_sec=perf_counter() - start,
            error_message=error_message,
        )


if __name__ == "__main__":
    main()
