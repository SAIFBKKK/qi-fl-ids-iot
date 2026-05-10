"""True Flower runtime for P8 FedAvg + QGA L1.

This module intentionally lives next to the P8 QGA code instead of replacing
the existing in-process adapters. It starts a real Flower server and real
Flower clients while applying the frozen QGA mask to all client train/val data.
The global test holdout is loaded only on the server during final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import flwr as fl
import numpy as np
import torch
from flwr.common import FitIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg

from fl_l1.client_data import ClientArrays, binary_counts, load_client_npz
from fl_l1.evaluation import finalize_test_metrics, tune_threshold_on_validation
from fl_l1.round_logger import ConsoleLogger, RoundLogger, format_round_console_line
from fl_l1.scenario_loader import L1Scenario, alpha_dir, load_l1_scenario, rel, write_json
from fl_l1_flower.communication import model_size_bytes, round_bandwidth
from fl_l1_flower.metrics import aggregate_evaluate_metrics, aggregate_fit_metrics, client_metrics_row
from fl_l1_flower.runtime import assert_port_available, configured_address, latest_run_id as _unused_latest_run_id
from fl_l1_flower.task import (
    build_model,
    client_fit_metrics,
    evaluate_arrays,
    get_parameters,
    parameter_payload_size,
    select_device,
    set_parameters,
    train_local,
)
from models.metrics import classification_report_dict
from qga.config import load_config, load_json, repo_path
from qga.feature_mask import apply_feature_mask, load_feature_names, load_latest_mask
from qga.fedavg_adapter import _comparison_with_p5
from qga.plotting import plot_binary_adapter_figures


@dataclass(frozen=True)
class QGAFlowerRunPaths:
    run_id: str
    scenario_dir: Path
    run_dir: Path
    checkpoints_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    latest_run_path: Path


@dataclass(frozen=True)
class QGAFlowerClientData:
    client_id: str
    train: ClientArrays
    val: ClientArrays
    expected_train_samples: int
    expected_val_samples: int


def _make_run_id() -> str:
    from datetime import datetime

    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_confusion_matrix(path: Path, metrics: dict[str, Any]) -> None:
    _write_csv(
        path,
        [
            {"label": "true_normal", "pred_normal": int(metrics["TN"]), "pred_attack": int(metrics["FP"])},
            {"label": "true_attack", "pred_normal": int(metrics["FN"]), "pred_attack": int(metrics["TP"])},
        ],
    )


def _mask_arrays(arrays: ClientArrays, mask: np.ndarray) -> ClientArrays:
    return ClientArrays(
        X=apply_feature_mask(arrays.X, mask).astype(np.float32, copy=False),
        y=arrays.y,
        label_id_original=arrays.label_id_original,
        row_id=arrays.row_id,
    )


def _final_selected_mask_dir(config: dict[str, Any]) -> Path:
    return repo_path(config, "outputs.qga_dir") / "final_selected_mask"


def _decision_features_count(decision: dict[str, Any], selected_mask_id: str | None) -> int | None:
    for key in ("features_count", "selected_features_count"):
        if key in decision:
            return int(decision[key])
    for row in decision.get("ranking", []):
        if not isinstance(row, dict):
            continue
        if selected_mask_id is None or row.get("mask_id") == selected_mask_id:
            if "features_count" in row:
                return int(row["features_count"])
    return None


def _load_final_selected_mask(config: dict[str, Any]) -> dict[str, Any]:
    final_dir = _final_selected_mask_dir(config)
    mask_path = final_dir / "feature_mask.json"
    selected_features_path = final_dir / "selected_features.json"
    decision_path = final_dir / "selection_decision.json"
    missing = [path for path in (mask_path, selected_features_path, decision_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "final_selected_mask is incomplete; missing: "
            + ", ".join(path.as_posix() for path in missing)
        )
    payload = load_json(mask_path)
    selected_features_payload = load_json(selected_features_path)
    decision = load_json(decision_path)
    if "mask" not in payload:
        raise ValueError(f"mask artifact does not contain a mask field: {mask_path}")
    mask = np.asarray(payload["mask"], dtype=np.int8)
    selected_count = int(mask.sum())
    selected_mask_id = str(decision.get("selected_mask_id") or payload.get("mask_id") or "")
    if selected_mask_id != "conservative_seed_42":
        raise ValueError(
            "P8 calibrated Flower runtime expected selected_mask_id=conservative_seed_42, "
            f"got {selected_mask_id!r}"
        )
    payload_mask_id = str(payload.get("mask_id", selected_mask_id))
    if payload_mask_id != selected_mask_id:
        raise ValueError(
            f"final_selected_mask mismatch: feature_mask mask_id={payload_mask_id!r}, "
            f"selection_decision selected_mask_id={selected_mask_id!r}"
        )
    expected_count = _decision_features_count(decision, selected_mask_id)
    if expected_count is not None and int(expected_count) != selected_count:
        raise ValueError(
            "final_selected_mask feature count mismatch: "
            f"selection_decision={expected_count}, feature_mask={selected_count}"
        )
    if int(payload.get("selected_features_count", selected_count)) != selected_count:
        raise ValueError(
            "final_selected_mask feature_mask selected_features_count does not match mask sum"
        )
    selected_features = payload.get("selected_features") or selected_features_payload.get("selected_features")
    if selected_features is not None and len(selected_features) != selected_count:
        raise ValueError(
            "final_selected_mask selected_features length does not match selected feature count"
        )
    payload = {
        **payload,
        "mask_id": selected_mask_id,
        "selected_features": selected_features or payload.get("selected_features", []),
        "selected_features_count": selected_count,
        "selected_mask_source": "final_selected_mask",
        "calibration_decision_used": True,
        "selection_decision": decision,
        "selection_decision_path": decision_path.as_posix(),
        "feature_mask_path": mask_path.as_posix(),
        "selected_features_path": selected_features_path.as_posix(),
    }
    return {
        "summary": decision,
        "payload": payload,
        "mask": mask,
        "metadata": {
            "selected_mask_id": selected_mask_id,
            "selected_mask_source": "final_selected_mask",
            "calibration_decision_used": True,
            "feature_mask_path": mask_path.as_posix(),
            "selected_features_path": selected_features_path.as_posix(),
            "selection_decision_path": decision_path.as_posix(),
        },
    }


def _assert_full_run_uses_final_mask(config: dict[str, Any], *, mask_path: str | Path | None, mode: str) -> None:
    if mode != "full" or mask_path is None:
        return
    final_mask_path = _final_selected_mask_dir(config) / "feature_mask.json"
    if final_mask_path.exists() and Path(mask_path).resolve() != final_mask_path.resolve():
        raise ValueError(
            "P8 FedAvg+QGA Flower full runs must use final_selected_mask. "
            f"Got explicit mask_path={mask_path}; expected {final_mask_path.as_posix()}"
        )


def load_mask_info(
    config: dict[str, Any],
    mask_path: str | Path | None = None,
    *,
    mask_source: str = "final_selected_mask",
) -> dict[str, Any]:
    if mask_path is None:
        if mask_source == "final_selected_mask":
            final_dir = _final_selected_mask_dir(config)
            if final_dir.exists():
                return _load_final_selected_mask(config)
            latest = load_latest_mask(repo_path(config, "outputs.qga_dir"))
            latest["metadata"] = {
                "selected_mask_id": latest["payload"].get("mask_id"),
                "selected_mask_source": "latest_qga_run",
                "calibration_decision_used": False,
            }
            return latest
        if mask_source == "latest_qga_run":
            final_dir = _final_selected_mask_dir(config)
            if final_dir.exists():
                raise ValueError(
                    "final_selected_mask exists; P8 Flower runtime must use it instead of latest_qga_run"
                )
            latest = load_latest_mask(repo_path(config, "outputs.qga_dir"))
            latest["metadata"] = {
                "selected_mask_id": latest["payload"].get("mask_id"),
                "selected_mask_source": "latest_qga_run",
                "calibration_decision_used": False,
            }
            return latest
        raise ValueError(f"unsupported mask_source: {mask_source}")
    payload = load_json(Path(mask_path))
    if "mask" not in payload:
        raise ValueError(f"mask artifact does not contain a mask field: {mask_path}")
    payload = {
        **payload,
        "selected_mask_source": "explicit_mask_path",
        "calibration_decision_used": False,
        "feature_mask_path": str(mask_path),
    }
    return {
        "summary": {},
        "payload": payload,
        "mask": np.asarray(payload["mask"], dtype=np.int8),
        "metadata": {
            "selected_mask_id": payload.get("mask_id"),
            "selected_mask_source": "explicit_mask_path",
            "calibration_decision_used": False,
            "feature_mask_path": str(mask_path),
        },
    }


def build_qga_flower_config(config: dict[str, Any], *, selected_count: int, alpha: float, clients: int, rounds: int) -> dict[str, Any]:
    """Build a P5.2-compatible runtime config from the P8 QGA config."""

    return {
        "project_root": config.get("project_root", "."),
        "final_experiment_dir": config["final_experiment_dir"],
        "inputs": {
            "partitions_root": config["inputs"]["l1_partitions_root"],
            "global_test_npz": config["inputs"]["test_npz"],
            "centralized_l1_metrics": "experiments/qi-fl-ids-iot-final/outputs/centralized_l1/artifacts/metrics_test.json",
        },
        "scenario": {
            "alpha": float(alpha),
            "clients": int(clients),
            "rounds": int(rounds),
            "default_alpha": float(alpha),
            "default_k": int(clients),
            "alphas": [float(alpha)],
            "clients_list": [int(clients)],
        },
        "model": {
            "name": "QGAFedAvgFlowerL1MLP",
            "input_dim": int(selected_count),
            "hidden_layers": [128, 64],
            "output_dim": 2,
            "dropout": 0.2,
            "activation": "relu",
        },
        "training": {
            "seed": int(config["qga"]["seed"]),
            "device": "auto",
            "batch_size": 512,
            "local_epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "threshold": {
            "start": 0.05,
            "stop": 0.95,
            "step": 0.05,
            "primary_objective": "f1_attack",
        },
        "flower": {
            "strategy": "FedAvg",
            "fraction_fit": float(config.get("flower", {}).get("fraction_fit", 1.0)),
            "fraction_evaluate": float(config.get("flower", {}).get("fraction_evaluate", 1.0)),
            "min_fit_clients": int(clients),
            "min_evaluate_clients": int(clients),
            "min_available_clients": int(clients),
            "address": str(config.get("flower", {}).get("address", "127.0.0.1:8083")),
        },
        "outputs": {
            "run_dir": config["outputs"]["qga_fedavg_flower_dir"],
            "figures_dir": f"{config['outputs']['figures_dir']}/fedavg_flower_l1",
            "reports_dir": config["outputs"]["reports_dir"],
        },
    }


def load_qga_flower_scenario(config: dict[str, Any], repo_root: Path, *, alpha: float, clients: int) -> L1Scenario:
    bridge_config = {
        "inputs": {
            "partitions_root": config["inputs"]["l1_partitions_root"],
            "global_test_npz": config["inputs"]["test_npz"],
        },
        "scenario": {
            "default_alpha": float(alpha),
            "default_k": int(clients),
            "alphas": [float(alpha)],
            "clients": [int(clients)],
        },
    }
    return load_l1_scenario(bridge_config, repo_root, alpha=float(alpha), num_clients=int(clients))


def prepare_qga_flower_run_paths(
    *,
    config: dict[str, Any],
    repo_root: Path,
    alpha: float,
    clients: int,
    run_id: str | None = None,
    mark_latest: bool = True,
) -> QGAFlowerRunPaths:
    resolved_run_id = run_id or _make_run_id()
    scenario_dir = repo_path(config, "outputs.qga_fedavg_flower_dir") / alpha_dir(float(alpha)) / f"k{int(clients)}"
    run_dir = scenario_dir / "runs" / resolved_run_id
    checkpoints_dir = run_dir / "checkpoints"
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    for directory in [checkpoints_dir, artifacts_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    latest_run_path = scenario_dir / "latest_run.json"
    if mark_latest:
        latest_run_path.parent.mkdir(parents=True, exist_ok=True)
        latest_run_path.write_text(
            json.dumps(
                {
                    "run_id": resolved_run_id,
                    "run_dir": rel(run_dir, repo_root),
                    "logs_dir": rel(logs_dir, repo_root),
                    "latest_run_summary": rel(scenario_dir / "latest_run_summary.json", repo_root),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return QGAFlowerRunPaths(
        run_id=resolved_run_id,
        scenario_dir=scenario_dir,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        artifacts_dir=artifacts_dir,
        logs_dir=logs_dir,
        latest_run_path=latest_run_path,
    )


def latest_qga_flower_run_id(*, config: dict[str, Any], alpha: float, clients: int) -> str:
    latest_path = repo_path(config, "outputs.qga_fedavg_flower_dir") / alpha_dir(float(alpha)) / f"k{int(clients)}" / "latest_run.json"
    if not latest_path.exists():
        raise FileNotFoundError(f"No latest P8 QGA Flower run found at {latest_path}. Start the server first.")
    return str(json.loads(latest_path.read_text(encoding="utf-8"))["run_id"])


def load_qga_flower_client_data(
    scenario: L1Scenario,
    *,
    client_id: str,
    mask: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> QGAFlowerClientData:
    partition = next((item for item in scenario.clients if item.client_id == client_id), None)
    if partition is None:
        raise ValueError(f"unknown client_id={client_id}")
    train = _mask_arrays(load_client_npz(partition.train_npz, max_samples=max_samples, seed=seed), mask)
    val = _mask_arrays(load_client_npz(partition.val_npz, max_samples=max_samples, seed=seed + 10_000), mask)
    return QGAFlowerClientData(
        client_id=client_id,
        train=train,
        val=val,
        expected_train_samples=partition.train_samples,
        expected_val_samples=partition.val_samples,
    )


def concatenate_masked_validation_arrays(
    scenario: L1Scenario,
    *,
    mask: np.ndarray,
    max_samples_per_client: int | None,
    seed: int,
) -> ClientArrays:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    label_ids: list[np.ndarray] = []
    row_ids: list[np.ndarray] = []
    for index, partition in enumerate(scenario.clients, start=1):
        arrays = _mask_arrays(
            load_client_npz(partition.val_npz, max_samples=max_samples_per_client, seed=seed + 20_000 + index),
            mask,
        )
        xs.append(arrays.X)
        ys.append(arrays.y)
        label_ids.append(arrays.label_id_original)
        row_ids.append(arrays.row_id)
    return ClientArrays(
        X=np.concatenate(xs, axis=0),
        y=np.concatenate(ys, axis=0),
        label_id_original=np.concatenate(label_ids, axis=0),
        row_id=np.concatenate(row_ids, axis=0),
    )


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


class QGAFedAvgFlowerClient(fl.client.NumPyClient):
    """True Flower NumPyClient for P8 FedAvg + QGA."""

    def __init__(
        self,
        *,
        client_id: str,
        runtime_config: dict[str, Any],
        scenario: L1Scenario,
        mask: np.ndarray,
        logs_dir: Path,
        max_samples_per_client: int | None,
    ) -> None:
        self.client_id = client_id
        self.config = runtime_config
        self.logs_dir = logs_dir
        self.training_cfg = runtime_config["training"]
        self.device = select_device(str(self.training_cfg["device"]))
        _append_log(self.logs_dir / "flower_clients.log", f"{client_id} loading data | global_test_loaded=false")
        self.data = load_qga_flower_client_data(
            scenario,
            client_id=client_id,
            mask=mask,
            max_samples=max_samples_per_client,
            seed=int(self.training_cfg["seed"]),
        )
        _append_log(
            self.logs_dir / "flower_clients.log",
            f"{client_id} data loaded | train={self.data.train.num_samples} val={self.data.val.num_samples} global_test_loaded=false",
        )
        self.model = build_model(runtime_config["model"]).to(self.device)
        _append_log(
            self.logs_dir / "flower_clients.log",
            (
                f"ClientApp ready | client_id={client_id} qga_features={runtime_config['model']['input_dim']} "
                f"train={self.data.train.num_samples}/{self.data.expected_train_samples} "
                f"val={self.data.val.num_samples}/{self.data.expected_val_samples}"
            ),
        )

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return get_parameters(self.model)

    def fit(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        round_number = int(config.get("server_round", 0))
        _append_log(self.logs_dir / "flower_clients.log", f"{self.client_id} fit started | round={round_number}")
        set_parameters(self.model, parameters)
        train_result = train_local(
            model=self.model,
            arrays=self.data.train,
            batch_size=int(self.training_cfg["batch_size"]),
            local_epochs=int(self.training_cfg["local_epochs"]),
            learning_rate=float(self.training_cfg["learning_rate"]),
            weight_decay=float(self.training_cfg["weight_decay"]),
            device=self.device,
            seed=int(self.training_cfg["seed"]) + round_number,
        )
        updated = get_parameters(self.model)
        val_result = evaluate_arrays(
            model=self.model,
            arrays=self.data.val,
            batch_size=int(self.training_cfg["batch_size"]) * 4,
            device=self.device,
            seed=int(self.training_cfg["seed"]),
            threshold=0.5,
        )
        metrics = client_fit_metrics(
            client_id=self.client_id,
            round_number=round_number,
            train_arrays=self.data.train,
            val_arrays=self.data.val,
            train_loss=float(train_result["loss"]),
            val_result=val_result,
            fit_time_sec=float(train_result["fit_time_sec"]),
            payload_size=parameter_payload_size(updated),
        )
        _append_log(
            self.logs_dir / "flower_clients.log",
            f"{self.client_id} fit completed | round={round_number} loss={metrics['local_train_loss']:.4f} macro_f1={metrics['local_macro_f1']:.4f}",
        )
        return updated, self.data.train.num_samples, metrics

    def evaluate(self, parameters: list[np.ndarray], config: dict[str, Any]) -> tuple[float, int, dict[str, Any]]:
        round_number = int(config.get("server_round", 0))
        set_parameters(self.model, parameters)
        result = evaluate_arrays(
            model=self.model,
            arrays=self.data.val,
            batch_size=int(self.training_cfg["batch_size"]) * 4,
            device=self.device,
            seed=int(self.training_cfg["seed"]),
            threshold=0.5,
        )
        metrics = client_fit_metrics(
            client_id=self.client_id,
            round_number=round_number,
            train_arrays=self.data.train,
            val_arrays=self.data.val,
            train_loss=0.0,
            val_result=result,
            fit_time_sec=0.0,
            payload_size=parameter_payload_size(parameters),
        )
        return float(metrics["local_val_loss"]), self.data.val.num_samples, metrics


class QGAFedAvgFlowerStrategy(FedAvg):
    """Flower FedAvg strategy with QGA mask-aware server evaluation."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        runtime_config: dict[str, Any],
        repo_root: Path,
        scenario: L1Scenario,
        run_paths: QGAFlowerRunPaths,
        validation_arrays: ClientArrays,
        mask_payload: dict[str, Any],
        mask_metadata: dict[str, Any],
        max_samples_per_client: int | None,
        mode: str,
        runtime_mode: str,
        evaluate_test: bool = True,
    ) -> None:
        self.config = config
        self.runtime_config = runtime_config
        self.repo_root = repo_root
        self.scenario = scenario
        self.run_id = run_paths.run_id
        self.run_dir = run_paths.run_dir
        self.checkpoints_dir = run_paths.checkpoints_dir
        self.artifacts_dir = run_paths.artifacts_dir
        self.logs_dir = run_paths.logs_dir
        self.validation_arrays = validation_arrays
        self.mask_payload = mask_payload
        self.mask_metadata = mask_metadata
        self.mask = np.asarray(mask_payload["mask"], dtype=np.int8)
        self.max_samples_per_client = max_samples_per_client
        self.mode = mode
        self.runtime_mode = runtime_mode
        self.evaluate_test = bool(evaluate_test)
        self.training_cfg = runtime_config["training"]
        self.threshold_cfg = runtime_config["threshold"]
        self.device = select_device(str(self.training_cfg["device"]))
        self.round_logger = RoundLogger(self.artifacts_dir, self.logs_dir, reset=True)
        self.console = ConsoleLogger(self.logs_dir / "run_console.log", reset=True)
        (self.logs_dir / "flower_server.log").write_text("", encoding="utf-8")
        self.client_rows: list[dict[str, Any]] = []
        self.round_rows: list[dict[str, Any]] = []
        self.cumulative_bytes = 0
        self.best_macro_f1 = -1.0
        self.best_round = 0
        self.best_parameters: Parameters | None = None
        self.latest_parameters: Parameters | None = None
        self.round_start: dict[int, float] = {}
        initial_model = build_model(runtime_config["model"])
        initial_parameters = ndarrays_to_parameters(get_parameters(initial_model))
        flower_cfg = runtime_config["flower"]
        super().__init__(
            fraction_fit=float(flower_cfg["fraction_fit"]),
            fraction_evaluate=float(flower_cfg["fraction_evaluate"]),
            min_fit_clients=int(flower_cfg["min_fit_clients"]),
            min_evaluate_clients=int(flower_cfg["min_evaluate_clients"]),
            min_available_clients=int(flower_cfg["min_available_clients"]),
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        )
        self._server_log(
            f"Starting true Flower FedAvg+QGA L1 server | alpha={scenario.alpha} K={scenario.num_clients} mode={mode} features={runtime_config['model']['input_dim']}"
        )

    def _server_log(self, message: str) -> None:
        line = f"QGA Flower server | {message}"
        self.console.log(line)
        with (self.logs_dir / "flower_server.log").open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        self.round_start[int(server_round)] = perf_counter()
        self._server_log(f"Starting round {server_round}")
        configured = super().configure_fit(server_round, parameters, client_manager)
        return [
            (client_proxy, FitIns(parameters=fit_ins.parameters, config={**dict(fit_ins.config), "server_round": int(server_round)}))
            for client_proxy, fit_ins in configured
        ]

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        configured = super().configure_evaluate(server_round, parameters, client_manager)
        updated = []
        for client_proxy, evaluate_ins in configured:
            cfg = {**dict(evaluate_ins.config), "server_round": int(server_round)}
            updated.append((client_proxy, type(evaluate_ins)(parameters=evaluate_ins.parameters, config=cfg)))
        return updated

    def _evaluate_server_validation(self, parameters: Parameters, *, threshold: float, collect_probabilities: bool = False) -> dict[str, Any]:
        model = build_model(self.runtime_config["model"]).to(self.device)
        set_parameters(model, parameters_to_ndarrays(parameters))
        return evaluate_arrays(
            model=model,
            arrays=self.validation_arrays,
            batch_size=int(self.training_cfg["batch_size"]) * 4,
            device=self.device,
            seed=int(self.training_cfg["seed"]),
            threshold=threshold,
            collect_probabilities=collect_probabilities,
        )

    def aggregate_fit(self, server_round: int, results, failures):
        aggregation_start = perf_counter()
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        aggregation_time_sec = perf_counter() - aggregation_start
        if aggregated_parameters is None:
            return aggregated_parameters, aggregated_metrics
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        train_losses: list[float] = []
        train_weights: list[int] = []
        for _, fit_res in results:
            row = client_metrics_row(dict(fit_res.metrics or {}))
            self.round_logger.log_client(row)
            self.client_rows.append(row)
            train_losses.append(float(row["local_train_loss"]))
            train_weights.append(int(fit_res.num_examples))
            weight = float(fit_res.num_examples / total_examples) if total_examples else 0.0
            self.round_logger.log_aggregation_weight(
                {"round": int(server_round), "client_id": row["client_id"], "num_examples": int(fit_res.num_examples), "aggregation_weight": weight}
            )
            self._server_log(f"Client fit completed | round={server_round} client={row['client_id']} samples={fit_res.num_examples}")
        val_result = self._evaluate_server_validation(aggregated_parameters, threshold=0.5)
        val_metrics = val_result["metrics"]
        size_bytes = model_size_bytes(parameters_to_ndarrays(aggregated_parameters))
        bandwidth = round_bandwidth(
            model_size_bytes_value=size_bytes,
            num_clients=len(results),
            previous_cumulative_bytes=self.cumulative_bytes,
        )
        self.cumulative_bytes = int(bandwidth["cumulative_bytes"])
        round_time_sec = perf_counter() - self.round_start.pop(int(server_round), perf_counter())
        row = {
            "round": int(server_round),
            "alpha": float(self.scenario.alpha),
            "num_clients": int(self.scenario.num_clients),
            "train_loss_mean": float(np.average(train_losses, weights=train_weights)) if train_weights else 0.0,
            "val_loss_mean": float(val_metrics["loss"]),
            "accuracy": float(val_metrics["accuracy"]),
            "precision": float(val_metrics["precision_attack"]),
            "recall": float(val_metrics["recall_attack"]),
            "macro_f1": float(val_metrics["macro_f1"]),
            "weighted_f1": float(val_metrics["weighted_f1"]),
            "attack_precision": float(val_metrics["precision_attack"]),
            "attack_recall": float(val_metrics["recall_attack"]),
            "attack_f1": float(val_metrics["f1_attack"]),
            "FPR": float(val_metrics["FPR"]),
            "FNR": float(val_metrics["FNR"]),
            "TP": int(val_metrics["TP"]),
            "TN": int(val_metrics["TN"]),
            "FP": int(val_metrics["FP"]),
            "FN": int(val_metrics["FN"]),
            "round_time_sec": float(round_time_sec),
            "aggregation_time_sec": float(aggregation_time_sec),
            "model_size_bytes": int(size_bytes),
            "communication_upload_bytes": int(bandwidth["upload_bytes"]),
            "communication_download_bytes": int(bandwidth["download_bytes"]),
            "communication_total_bytes": int(bandwidth["total_bytes"]),
            "communication_cumulative_bytes": int(bandwidth["cumulative_bytes"]),
        }
        self.round_logger.log_round(row)
        self.round_logger.log_bandwidth({"round": int(server_round), **bandwidth})
        self.round_rows.append(row)
        self.console.log(format_round_console_line(row, current_round=int(server_round), total_rounds=int(self.runtime_config["scenario"]["rounds"])))
        self.latest_parameters = aggregated_parameters
        self._save_checkpoint("last_global_model.pth", aggregated_parameters, server_round, row)
        if float(row["macro_f1"]) > self.best_macro_f1:
            self.best_macro_f1 = float(row["macro_f1"])
            self.best_round = int(server_round)
            self.best_parameters = aggregated_parameters
            self._save_checkpoint("best_global_model.pth", aggregated_parameters, server_round, row)
        return aggregated_parameters, aggregated_metrics

    def _save_checkpoint(self, filename: str, parameters: Parameters, server_round: int, metrics: dict[str, Any]) -> None:
        model = build_model(self.runtime_config["model"])
        set_parameters(model, parameters_to_ndarrays(parameters))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "round": int(server_round),
                "selection_metric": "server_validation_macro_f1",
                "selection_metric_value": float(metrics["macro_f1"]),
                "test_used_for_selection": False,
                "qga_mask": self.mask_payload,
                "true_flower_runtime": True,
            },
            self.checkpoints_dir / filename,
        )

    def finalize(self) -> dict[str, Any]:
        selected = self.best_parameters or self.latest_parameters
        if selected is None:
            raise RuntimeError("No Flower parameters available to finalize QGA run")
        val_result = self._evaluate_server_validation(selected, threshold=0.5, collect_probabilities=True)
        threshold_payload, threshold_rows = tune_threshold_on_validation(
            val_result["y_true"],
            val_result["prob_attack"],
            start=float(self.threshold_cfg["start"]),
            stop=float(self.threshold_cfg["stop"]),
            step=float(self.threshold_cfg["step"]),
        )
        primary_threshold = float(threshold_payload["primary_threshold"])
        metrics_val = dict(threshold_payload["primary_validation_metrics"])
        metrics_val["selection_split"] = "server_validation"
        metrics_val["test_used_for_threshold"] = False
        model = build_model(self.runtime_config["model"]).to(self.device)
        set_parameters(model, parameters_to_ndarrays(selected))
        num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
        size_bytes = model_size_bytes(model.state_dict())
        metrics_test: dict[str, Any] = {}
        test_result: dict[str, Any] | None = None
        if self.evaluate_test:
            test_arrays = _mask_arrays(
                load_client_npz(
                    self.repo_root / self.config["inputs"]["test_npz"],
                    max_samples=self.max_samples_per_client if self.mode == "smoke" else None,
                    seed=int(self.training_cfg["seed"]) + 99_000,
                ),
                self.mask,
            )
            test_result = evaluate_arrays(
                model=model,
                arrays=test_arrays,
                batch_size=int(self.training_cfg["batch_size"]) * 4,
                device=self.device,
                seed=int(self.training_cfg["seed"]),
                threshold=primary_threshold,
                collect_probabilities=True,
            )
            metrics_test = finalize_test_metrics(test_result["y_true"], test_result["prob_attack"], primary_threshold)
            metrics_test["threshold"] = primary_threshold
            metrics_test["model_size_bytes"] = size_bytes
            metrics_test["num_parameters"] = num_parameters
        model_config = {
            **self.runtime_config["model"],
            "architecture": f"{self.runtime_config['model']['input_dim']} -> 128 -> 64 -> 2",
            "num_parameters": int(num_parameters),
            "model_size_bytes": int(size_bytes),
            "qga_selected_features_count": int(self.mask.sum()),
            "selected_mask_id": self.mask_metadata.get("selected_mask_id"),
            "selected_mask_source": self.mask_metadata.get("selected_mask_source"),
            "calibration_decision_used": bool(self.mask_metadata.get("calibration_decision_used")),
        }
        write_json(self.artifacts_dir / "model_config.json", model_config)
        write_json(self.artifacts_dir / "selected_features_reference.json", self.mask_payload)
        write_json(self.artifacts_dir / "metrics_val.json", metrics_val)
        if self.evaluate_test:
            write_json(self.artifacts_dir / "metrics_test.json", metrics_test)
        write_json(self.artifacts_dir / "threshold.json", threshold_payload)
        _write_csv(self.artifacts_dir / "threshold_sweep.csv", threshold_rows)
        confusion_source = metrics_test if self.evaluate_test else metrics_val
        _write_confusion_matrix(self.artifacts_dir / "confusion_matrix.csv", confusion_source)
        comparison: dict[str, Any] = {}
        if self.evaluate_test:
            write_json(self.artifacts_dir / "classification_report.json", metrics_test["classification_report"])
            comparison = _comparison_with_p5(self.config, alpha=float(self.scenario.alpha), clients=int(self.scenario.num_clients), test_metrics=metrics_test)
            write_json(self.artifacts_dir / "comparison_with_p5.json", comparison)
        figures_dir = repo_path(self.config, "outputs.figures_dir") / "fedavg_flower_l1" / alpha_dir(float(self.scenario.alpha)) / f"k{int(self.scenario.num_clients)}" / self.run_id
        figures = plot_binary_adapter_figures(
            metrics_rounds=self.round_rows,
            confusion_metrics=confusion_source,
            output_dir=figures_dir,
            prefix="p8_fedavg_qga_flower",
        )
        global_test_rows_expected: int | None = None
        if self.evaluate_test:
            with np.load(self.scenario.global_test_npz, allow_pickle=False) as npz:
                global_test_rows_expected = int(npz["y_binary"].shape[0])
        client_train_rows = {client.client_id: int(client.train_samples) for client in self.scenario.clients}
        client_val_rows = {client.client_id: int(client.val_samples) for client in self.scenario.clients}
        rounds_completed = len(self.round_rows)
        run_summary_path = self.artifacts_dir / "run_summary.json"
        run_manifest_path = self.artifacts_dir / "run_manifest.json"
        root_manifest_path = self.run_dir / "manifest.json"
        artifacts_pre = [
            self.run_dir / "manifest.json",
            self.checkpoints_dir / "best_global_model.pth",
            self.checkpoints_dir / "last_global_model.pth",
            *[path for path in self.artifacts_dir.iterdir() if path.is_file()],
            run_summary_path,
            run_manifest_path,
            self.logs_dir / "flower_server.log",
            self.logs_dir / "flower_clients.log",
            self.logs_dir / "run_console.log",
        ]
        artifacts = [rel(path, self.repo_root) for path in artifacts_pre if path.exists()]
        figure_paths = [rel(Path(path), self.repo_root) for path in figures]
        criteria = {
            "qga_mask_applied": True,
            "true_flower_runtime": True,
            "flower_runtime_true": True,
            "server_client_runtime_started": rounds_completed > 0,
            "p3_partitions_used": True,
            "global_test_holdout_protected": True,
            "test_sent_to_clients_false": True,
            "test_not_used_for_selection": True,
            "threshold_validation_only": True,
            "round_metrics_generated": (self.artifacts_dir / "metrics_rounds.csv").exists(),
            "client_metrics_generated": (self.artifacts_dir / "metrics_clients.csv").exists(),
            "bandwidth_metrics_generated": (self.artifacts_dir / "bandwidth_rounds.csv").exists(),
            "best_model_saved": (self.checkpoints_dir / "best_global_model.pth").exists(),
            "last_model_saved": (self.checkpoints_dir / "last_global_model.pth").exists(),
            "metrics_test_generated": (self.artifacts_dir / "metrics_test.json").exists(),
            "comparison_with_p5_generated": (self.artifacts_dir / "comparison_with_p5.json").exists(),
            "figures_generated": len(figure_paths) >= 3,
            "calibration_decision_used": bool(self.mask_metadata.get("calibration_decision_used")),
            "final_selected_mask_used": self.mask_metadata.get("selected_mask_source") == "final_selected_mask",
            "validation_only": not self.evaluate_test,
            "test_not_loaded_for_short_validation": not self.evaluate_test,
            "smoke_run_only": self.mode == "smoke",
            "full_run_completed": self.mode == "full" and rounds_completed == int(self.runtime_config["scenario"]["rounds"]),
        }
        warnings: list[str] = []
        if self.mode == "smoke":
            warnings.append("Smoke run uses sampled clients/test data; metrics have low scientific significance.")
        if not self.evaluate_test:
            warnings.append("Validation-only calibration run did not load or evaluate the global test holdout.")
        if metrics_test.get("roc_pr_warning"):
            warnings.append(str(metrics_test["roc_pr_warning"]))
        required_criteria = [
            "qga_mask_applied",
            "true_flower_runtime",
            "server_client_runtime_started",
            "p3_partitions_used",
            "global_test_holdout_protected",
            "test_sent_to_clients_false",
            "test_not_used_for_selection",
            "threshold_validation_only",
            "round_metrics_generated",
            "client_metrics_generated",
            "bandwidth_metrics_generated",
            "best_model_saved",
            "last_model_saved",
            "figures_generated",
        ]
        if self.evaluate_test:
            required_criteria.extend(["metrics_test_generated", "comparison_with_p5_generated"])
        else:
            required_criteria.append("test_not_loaded_for_short_validation")
        summary = {
            "accepted": all(bool(criteria[key]) for key in required_criteria),
            "phase": "P8",
            "method": "FedAvg + QGA",
            "runtime": self.runtime_mode,
            "true_flower_runtime": True,
            "flower_runtime": True,
            "flower_version": fl.__version__,
            "mode": self.mode,
            "run_id": self.run_id,
            "selected_mask_id": self.mask_metadata.get("selected_mask_id"),
            "selected_mask_source": self.mask_metadata.get("selected_mask_source"),
            "calibration_decision_used": bool(self.mask_metadata.get("calibration_decision_used")),
            "selected_features_count": int(self.mask.sum()),
            "selected_features": self.mask_payload["selected_features"],
            "input_dim_selected": int(self.mask.sum()),
            "scenario": {
                "alpha": float(self.scenario.alpha),
                "clients": int(self.scenario.num_clients),
                "client_ids": [client.client_id for client in self.scenario.clients],
                "rounds": int(self.runtime_config["scenario"]["rounds"]),
            },
            "qga": {
                "selected_mask_id": self.mask_metadata.get("selected_mask_id"),
                "selected_mask_source": self.mask_metadata.get("selected_mask_source"),
                "calibration_decision_used": bool(self.mask_metadata.get("calibration_decision_used")),
                "selected_features_count": int(self.mask.sum()),
                "selected_indices": self.mask_payload["selected_indices"],
                "selected_features": self.mask_payload["selected_features"],
                "feature_mask_path": rel(Path(self.mask_metadata.get("feature_mask_path", "")), self.repo_root)
                if self.mask_metadata.get("feature_mask_path")
                else "",
                "selection_decision_path": rel(Path(self.mask_metadata.get("selection_decision_path", "")), self.repo_root)
                if self.mask_metadata.get("selection_decision_path")
                else "",
                "test_used_for_selection": False,
            },
            "dataset": {
                "input_dim_original": 28,
                "input_dim_selected": int(self.mask.sum()),
                "train_rows_total": int(sum(client_train_rows.values())),
                "val_rows_total": int(sum(client_val_rows.values())),
                "test_rows": int(metrics_test["support_total"]) if self.evaluate_test else None,
                "global_test_rows_expected": global_test_rows_expected,
                "client_train_rows": client_train_rows,
                "client_val_rows": client_val_rows,
                "global_test_holdout": rel(self.scenario.global_test_npz, self.repo_root),
                "test_sent_to_clients": False,
            },
            "model": model_config,
            "training": {
                "strategy": "FedAvg",
                "framework": "Flower",
                "rounds_configured": int(self.runtime_config["scenario"]["rounds"]),
                "rounds_completed": rounds_completed,
                "best_round": int(self.best_round),
                "selection_metric": "server_validation_macro_f1",
            },
            "threshold": threshold_payload,
            "validation": {"metrics": metrics_val, "selection_split": "server_validation", "test_used_for_threshold": False},
            "test": {
                "metrics": metrics_test,
                "global_holdout": rel(self.scenario.global_test_npz, self.repo_root),
                "evaluated": self.evaluate_test,
                "used_for_selection": False,
                "loaded": self.evaluate_test,
            },
            "comparison_with_p5": comparison,
            "communication": {
                "model_size_bytes": int(size_bytes),
                "communication_cumulative_bytes": int(self.cumulative_bytes),
            },
            "artifacts": artifacts,
            "figures": figure_paths,
            "criteria": criteria,
            "warnings": warnings,
            "errors": [],
        }
        manifest = {
            "run_id": self.run_id,
            "accepted": summary["accepted"],
            "true_flower_runtime": True,
            "test_sent_to_clients": False,
            "run_summary": rel(self.artifacts_dir / "run_summary.json", self.repo_root),
            "global_test_holdout": rel(self.scenario.global_test_npz, self.repo_root),
            "artifacts_dir": rel(self.artifacts_dir, self.repo_root),
            "logs_dir": rel(self.logs_dir, self.repo_root),
        }
        write_json(run_summary_path, summary)
        write_json(run_manifest_path, manifest)
        write_json(root_manifest_path, manifest)
        latest_summary_path = self.run_dir.parents[1] / "latest_run_summary.json"
        write_json(latest_summary_path, summary)
        summary["artifacts"] = sorted(set(summary["artifacts"] + [rel(run_summary_path, self.repo_root), rel(run_manifest_path, self.repo_root), rel(root_manifest_path, self.repo_root), rel(latest_summary_path, self.repo_root)]))
        write_json(run_summary_path, summary)
        write_json(latest_summary_path, summary)
        if self.evaluate_test:
            self._server_log(f"Final evaluation on global test holdout | macro_f1={metrics_test['macro_f1']:.4f}")
        else:
            self._server_log(f"Validation-only calibration finalized | val_macro_f1={metrics_val['macro_f1']:.4f}")
        return summary


def build_qga_flower_strategy(
    *,
    config: dict[str, Any],
    repo_root: Path,
    alpha: float,
    clients: int,
    rounds: int,
    max_samples_per_client: int | None,
    mode: str,
    run_id: str | None,
    runtime_mode: str,
    mask_path: str | Path | None = None,
    mask_source: str = "final_selected_mask",
    evaluate_test: bool = True,
) -> QGAFedAvgFlowerStrategy:
    _assert_full_run_uses_final_mask(config, mask_path=mask_path, mode=mode)
    mask_info = load_mask_info(config, mask_path, mask_source=mask_source)
    mask = mask_info["mask"]
    runtime_config = build_qga_flower_config(
        config,
        selected_count=int(mask.sum()),
        alpha=alpha,
        clients=clients,
        rounds=rounds,
    )
    runtime_config["scenario"]["rounds"] = int(rounds)
    scenario = load_qga_flower_scenario(config, repo_root, alpha=alpha, clients=clients)
    run_paths = prepare_qga_flower_run_paths(config=config, repo_root=repo_root, alpha=alpha, clients=clients, run_id=run_id)
    validation_arrays = concatenate_masked_validation_arrays(
        scenario,
        mask=mask,
        max_samples_per_client=max_samples_per_client if mode == "smoke" else None,
        seed=int(runtime_config["training"]["seed"]),
    )
    return QGAFedAvgFlowerStrategy(
        config=config,
        runtime_config=runtime_config,
        repo_root=repo_root,
        scenario=scenario,
        run_paths=run_paths,
        validation_arrays=validation_arrays,
        mask_payload=mask_info["payload"],
        mask_metadata=mask_info.get("metadata", {}),
        max_samples_per_client=max_samples_per_client if mode == "smoke" else None,
        mode=mode,
        runtime_mode=runtime_mode,
        evaluate_test=evaluate_test,
    )


def start_qga_flower_server(
    *,
    config: dict[str, Any],
    repo_root: Path,
    alpha: float,
    clients: int,
    rounds: int,
    max_samples_per_client: int | None,
    mode: str,
    address: str,
    run_id: str | None = None,
    runtime_mode: str = "manual",
    check_port: bool = True,
    mask_path: str | Path | None = None,
    mask_source: str = "final_selected_mask",
    evaluate_test: bool = True,
) -> dict[str, Any]:
    if check_port:
        assert_port_available(address)
    strategy = build_qga_flower_strategy(
        config=config,
        repo_root=repo_root,
        alpha=alpha,
        clients=clients,
        rounds=rounds,
        max_samples_per_client=max_samples_per_client,
        mode=mode,
        run_id=run_id,
        runtime_mode=runtime_mode,
        mask_path=mask_path,
        mask_source=mask_source,
        evaluate_test=evaluate_test,
    )
    strategy._server_log(f"server starting | address={address} run_id={strategy.run_id}")
    strategy._server_log(f"server listening on address={address}")
    strategy._server_log(f"server waiting for clients | min_available={clients}")
    fl.server.start_server(
        server_address=address,
        config=ServerConfig(num_rounds=int(rounds)),
        strategy=strategy,
    )
    summary = strategy.finalize()
    strategy._server_log("server finished")
    return summary


def start_qga_flower_client(
    *,
    config: dict[str, Any],
    repo_root: Path,
    client_id: str,
    alpha: float,
    clients: int,
    address: str,
    max_samples_per_client: int | None,
    run_id: str | None,
    mode: str,
    mask_path: str | Path | None = None,
    mask_source: str = "final_selected_mask",
) -> None:
    _assert_full_run_uses_final_mask(config, mask_path=mask_path, mode=mode)
    mask_info = load_mask_info(config, mask_path, mask_source=mask_source)
    mask = mask_info["mask"]
    scenario = load_qga_flower_scenario(config, repo_root, alpha=alpha, clients=clients)
    resolved_run_id = run_id or latest_qga_flower_run_id(config=config, alpha=alpha, clients=clients)
    run_paths = prepare_qga_flower_run_paths(
        config=config,
        repo_root=repo_root,
        alpha=alpha,
        clients=clients,
        run_id=resolved_run_id,
        mark_latest=False,
    )
    runtime_config = build_qga_flower_config(
        config,
        selected_count=int(mask.sum()),
        alpha=alpha,
        clients=clients,
        rounds=int(config.get("fedavg_eval", {}).get("rounds", 30)),
    )
    _append_log(run_paths.logs_dir / "flower_clients.log", f"{client_id} process starting | address={address} run_id={resolved_run_id}")
    client = QGAFedAvgFlowerClient(
        client_id=client_id,
        runtime_config=runtime_config,
        scenario=scenario,
        mask=mask,
        logs_dir=run_paths.logs_dir,
        max_samples_per_client=max_samples_per_client if mode == "smoke" else None,
    )
    _append_log(run_paths.logs_dir / "flower_clients.log", f"{client_id} connecting to server | address={address}")
    try:
        fl.client.start_client(server_address=address, client=client.to_client())
        _append_log(run_paths.logs_dir / "flower_clients.log", f"{client_id} connection finished cleanly")
    except BaseException as exc:
        _append_log(run_paths.logs_dir / "flower_clients.log", f"{client_id} connection failed | error={exc}")
        raise


def run_qga_flower_smoke_subprocess(
    *,
    config_path: Path,
    alpha: float,
    clients: int,
    rounds: int,
    max_samples_per_client: int,
    address: str,
    timeout_sec: int = 600,
    mask_path: str | Path | None = None,
    mask_source: str = "final_selected_mask",
    evaluate_test: bool = True,
    mode: str = "smoke",
) -> dict[str, Any]:
    repo_root = Path.cwd().resolve()
    config = load_config(config_path)
    assert_port_available(address)
    run_id = _make_run_id()
    prepare_qga_flower_run_paths(config=config, repo_root=repo_root, alpha=alpha, clients=clients, run_id=run_id)
    scripts_dir = repo_root / "experiments" / "qi-fl-ids-iot-final" / "src" / "scripts"
    server_script = scripts_dir / "08_start_qga_fedavg_flower_server.py"
    client_script = scripts_dir / "08_start_qga_fedavg_flower_client.py"
    common = [
        "--config",
        str(config_path),
        "--alpha",
        str(alpha),
        "--clients",
        str(clients),
        "--address",
        address,
        "--run-id",
        run_id,
        "--mode",
        mode,
        "--max-samples-per-client",
        str(max_samples_per_client),
    ]
    if mask_path is not None:
        common.extend(["--mask-path", str(mask_path)])
    else:
        common.extend(["--mask-source", str(mask_source)])
    server_cmd = [sys.executable, str(server_script), *common, "--rounds", str(rounds), "--runtime-label", "subprocess"]
    if not evaluate_test:
        server_cmd.append("--validation-only")
    processes: list[tuple[str, subprocess.Popen]] = []
    handles = []
    run_paths = prepare_qga_flower_run_paths(
        config=config,
        repo_root=repo_root,
        alpha=alpha,
        clients=clients,
        run_id=run_id,
        mark_latest=False,
    )

    def start(name: str, command: list[str]) -> None:
        log_path = run_paths.logs_dir / f"{name}_stdout.log"
        handle = log_path.open("w", encoding="utf-8")
        handles.append(handle)
        processes.append(
            (
                name,
                subprocess.Popen(
                    command,
                    cwd=str(repo_root),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                ),
            )
        )

    try:
        start("server", server_cmd)
        time.sleep(5.0)
        for client_index in range(1, int(clients) + 1):
            client_id = f"client_{client_index}"
            client_cmd = [sys.executable, str(client_script), *common, "--client-id", client_id]
            start(client_id, client_cmd)
        deadline = time.monotonic() + float(timeout_sec)
        for name, process in processes:
            remaining = max(1.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"{name} did not finish within {timeout_sec}s") from exc
        failures = [(name, process.returncode) for name, process in processes if process.returncode != 0]
        if failures:
            raise RuntimeError(f"P8 QGA Flower subprocess failure(s): {failures}")
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for handle in handles:
            handle.close()
    summary_path = run_paths.artifacts_dir / "run_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"P8 QGA Flower smoke did not produce {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))
