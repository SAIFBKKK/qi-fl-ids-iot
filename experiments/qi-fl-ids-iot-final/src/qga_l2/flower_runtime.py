"""True Flower runtime for P8-b L2 FedAvg + QGA."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import flwr as fl
import numpy as np
import torch
from flwr.common import FitIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg

from fl_hierarchical.communication import model_size_bytes, round_bandwidth
from fl_hierarchical.data import make_dataloader
from fl_hierarchical.metrics import classification_report, multiclass_metrics, one_vs_rest_rows, write_rows
from fl_hierarchical.models import build_model, get_parameters, set_parameters
from fl_hierarchical.runtime import assert_port_available, make_run_id
from fl_hierarchical.strategy import client_metrics_payload, evaluate_arrays, select_device, train_local
from qga_l2.config import alpha_dir, load_config, load_json, rel, repo_path, write_json
from qga_l2.data import concatenate_masked_validation_arrays, load_masked_client_data, load_masked_global_arrays
from qga_l2.feature_mask import load_final_mask


def _macro_recall(metrics: dict[str, Any]) -> float:
    return float(metrics.get("macro_recall", metrics.get("recall_macro", 0.0)))


def _macro_precision(metrics: dict[str, Any]) -> float:
    return float(metrics.get("macro_precision", metrics.get("precision_macro", 0.0)))


def _macro_fpr(metrics: dict[str, Any]) -> float:
    return float(metrics.get("macro_fpr", metrics.get("FPR_macro", 0.0)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def scenario_dir(config: dict[str, Any], *, alpha: float, clients: int) -> Path:
    return repo_path(config, "outputs.qga_l2_flower_dir") / alpha_dir(float(alpha)) / f"k{int(clients)}"


def prepare_run_paths(config: dict[str, Any], *, alpha: float, clients: int, run_id: str | None = None, mark_latest: bool = True) -> dict[str, Path | str]:
    resolved = run_id or make_run_id()
    base = scenario_dir(config, alpha=alpha, clients=clients)
    run_dir = base / "runs" / resolved
    paths = {
        "run_id": resolved,
        "scenario_dir": base,
        "run_dir": run_dir,
        "checkpoints_dir": run_dir / "checkpoints",
        "artifacts_dir": run_dir / "artifacts",
        "logs_dir": run_dir / "logs",
    }
    for key in ("checkpoints_dir", "artifacts_dir", "logs_dir"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)
    if mark_latest:
        write_json(
            base / "latest_run.json",
            {
                "run_id": resolved,
                "run_dir": rel(run_dir, repo_path(config)),
                "latest_run_summary": rel(base / "latest_run_summary.json", repo_path(config)),
            },
        )
    return paths


def latest_run_id(config: dict[str, Any], *, alpha: float, clients: int) -> str:
    path = scenario_dir(config, alpha=alpha, clients=clients) / "latest_run.json"
    if not path.exists():
        raise FileNotFoundError(f"No latest P8-b L2 Flower run found at {path}")
    return str(load_json(path)["run_id"])


def _load_mask(config: dict[str, Any], mask_path: str | Path | None = None) -> dict[str, Any]:
    if mask_path is None:
        return load_final_mask(config)
    payload = load_json(mask_path)
    return {
        "payload": {
            **payload,
            "selected_mask_source": "explicit_mask_path",
            "calibration_decision_used": False,
            "feature_mask_path": str(mask_path),
        },
        "decision": {},
        "mask": np.asarray(payload["mask"], dtype=np.int8),
    }


class QGAL2FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        client_id: str,
        alpha: float,
        clients: int,
        mask: np.ndarray,
        run_id: str,
        max_samples_per_client: int | None,
    ) -> None:
        self.config = config
        self.client_id = client_id
        self.mask = mask
        self.repo_root = repo_path(config)
        self.paths = prepare_run_paths(config, alpha=alpha, clients=clients, run_id=run_id, mark_latest=False)
        self.logs_dir = Path(self.paths["logs_dir"])
        self.device = select_device(str(config["training"]["device"]))
        self.data = load_masked_client_data(
            config,
            self.repo_root,
            alpha=alpha,
            clients=clients,
            client_id=client_id,
            mask=mask,
            max_samples_per_client=max_samples_per_client,
        )
        model_cfg = dict(config["model"])
        model_cfg["input_dim"] = int(mask.sum())
        self.model = build_model(model_cfg, output_dim=8).to(self.device)
        _append_log(self.logs_dir / "flower_clients.log", f"ClientApp ready | client_id={client_id} train={self.data.train.num_samples} val={self.data.val.num_samples} global_test_loaded=false")

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return get_parameters(self.model)

    def fit(self, parameters: list[np.ndarray], config: dict[str, Any]):
        round_number = int(config.get("server_round", 0))
        _append_log(self.logs_dir / "flower_clients.log", f"{self.client_id} fit started | round={round_number}")
        set_parameters(self.model, parameters)
        result = train_local(
            model=self.model,
            arrays=self.data.train,
            batch_size=int(self.config["training"]["batch_size"]),
            local_epochs=int(self.config["training"]["local_epochs"]),
            learning_rate=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"]["weight_decay"]),
            device=self.device,
            seed=int(self.config["training"]["seed"]) + round_number,
            num_classes=8,
            use_class_weights=bool(self.config["training"].get("use_class_weights", True)),
        )
        updated = get_parameters(self.model)
        val = evaluate_arrays(
            model=self.model,
            arrays=self.data.val,
            batch_size=int(self.config["training"]["batch_size"]) * 4,
            device=self.device,
            seed=int(self.config["training"]["seed"]),
            class_names=[str(i) for i in range(8)],
        )
        metrics = client_metrics_payload(
            client_id=self.client_id,
            round_number=round_number,
            train_arrays=self.data.train,
            val_arrays=self.data.val,
            train_loss=float(result["loss"]),
            val_result=val,
            fit_time_sec=float(result["fit_time_sec"]),
            payload_size=sum(array.nbytes for array in updated),
        )
        _append_log(self.logs_dir / "flower_clients.log", f"{self.client_id} fit completed | round={round_number} macro_f1={metrics['local_macro_f1']:.4f}")
        return updated, self.data.train.num_samples, metrics

    def evaluate(self, parameters: list[np.ndarray], config: dict[str, Any]):
        set_parameters(self.model, parameters)
        val = evaluate_arrays(
            model=self.model,
            arrays=self.data.val,
            batch_size=int(self.config["training"]["batch_size"]) * 4,
            device=self.device,
            seed=int(self.config["training"]["seed"]),
            class_names=[str(i) for i in range(8)],
        )
        metrics = client_metrics_payload(
            client_id=self.client_id,
            round_number=int(config.get("server_round", 0)),
            train_arrays=self.data.train,
            val_arrays=self.data.val,
            train_loss=0.0,
            val_result=val,
            fit_time_sec=0.0,
            payload_size=sum(array.nbytes for array in parameters),
        )
        return float(metrics["local_val_loss"]), self.data.val.num_samples, metrics


class QGAL2FedAvgStrategy(FedAvg):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        alpha: float,
        clients: int,
        rounds: int,
        mask_info: dict[str, Any],
        run_paths: dict[str, Path | str],
        max_samples_per_client: int | None,
        mode: str,
        evaluate_test: bool,
    ) -> None:
        self.config = config
        self.alpha = float(alpha)
        self.clients = int(clients)
        self.rounds = int(rounds)
        self.mask_info = mask_info
        self.mask = np.asarray(mask_info["mask"], dtype=np.int8)
        self.run_id = str(run_paths["run_id"])
        self.run_dir = Path(run_paths["run_dir"])
        self.checkpoints_dir = Path(run_paths["checkpoints_dir"])
        self.artifacts_dir = Path(run_paths["artifacts_dir"])
        self.logs_dir = Path(run_paths["logs_dir"])
        self.mode = mode
        self.evaluate_test = evaluate_test
        self.repo_root = repo_path(config)
        self.device = select_device(str(config["training"]["device"]))
        self.round_rows: list[dict[str, Any]] = []
        self.client_rows: list[dict[str, Any]] = []
        self.best_macro_f1 = -1.0
        self.best_round = 0
        self.best_parameters: Parameters | None = None
        self.latest_parameters: Parameters | None = None
        self.cumulative_bytes = 0
        model_cfg = dict(config["model"])
        model_cfg["input_dim"] = int(self.mask.sum())
        model = build_model(model_cfg, output_dim=8)
        initial = ndarrays_to_parameters(get_parameters(model))
        super().__init__(
            fraction_fit=float(config["flower"]["fraction_fit"]),
            fraction_evaluate=float(config["flower"]["fraction_evaluate"]),
            min_fit_clients=int(clients),
            min_evaluate_clients=int(clients),
            min_available_clients=int(clients),
            initial_parameters=initial,
        )
        _append_log(self.logs_dir / "flower_server.log", f"Starting P8-b L2 QGA Flower server | alpha={alpha} K={clients} mode={mode} features={int(self.mask.sum())}")

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        config = {"server_round": int(server_round)}
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results, failures):
        start = perf_counter()
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self.latest_parameters = aggregated
        upload = 0
        train_loss = []
        for _, fit_res in results:
            upload += int(fit_res.metrics.get("upload_bytes", 0) or fit_res.metrics.get("payload_size", 0) or 0)
            train_loss.append(float(fit_res.metrics.get("local_train_loss", 0.0)))
            self.client_rows.append({"round": server_round, **dict(fit_res.metrics)})
        size = int(model_size_bytes(build_model({"input_dim": int(self.mask.sum()), "hidden_layers": [128, 64], "dropout": 0.2}, output_dim=8).state_dict()))
        bw = round_bandwidth(model_size_bytes_value=size, num_clients=self.clients)
        self.cumulative_bytes += int(bw["total_bytes"])
        self.round_rows.append(
            {
                "round": int(server_round),
                "train_loss_mean": float(np.mean(train_loss)) if train_loss else 0.0,
                "aggregation_time_sec": perf_counter() - start,
                "model_size_bytes": size,
                "communication_total_bytes": int(bw["total_bytes"]),
                "communication_cumulative_bytes": int(self.cumulative_bytes),
            }
        )
        return aggregated, metrics

    def evaluate(self, server_round: int, parameters: Parameters):
        model_cfg = dict(self.config["model"])
        model_cfg["input_dim"] = int(self.mask.sum())
        model = build_model(model_cfg, output_dim=8).to(self.device)
        set_parameters(model, parameters_to_ndarrays(parameters))
        val_arrays = concatenate_masked_validation_arrays(
            self.config,
            self.repo_root,
            alpha=self.alpha,
            clients=self.clients,
            mask=self.mask,
            max_samples_per_client=int(self.config["execution"]["smoke_max_samples_per_client"]) if self.mode == "smoke" else None,
        )
        result = evaluate_arrays(
            model=model,
            arrays=val_arrays,
            batch_size=int(self.config["training"]["batch_size"]) * 4,
            device=self.device,
            seed=int(self.config["training"]["seed"]),
            class_names=[str(i) for i in range(8)],
        )
        metrics = result["metrics"]
        if self.round_rows:
            self.round_rows[-1].update(
                {
                    "val_loss_mean": float(metrics.get("loss", 0.0)),
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "weighted_f1": float(metrics.get("weighted_f1", 0.0)),
                    "macro_recall": _macro_recall(metrics),
                    "macro_precision": _macro_precision(metrics),
                    "macro_fpr": _macro_fpr(metrics),
                }
            )
        if float(metrics["macro_f1"]) > self.best_macro_f1:
            self.best_macro_f1 = float(metrics["macro_f1"])
            self.best_round = int(server_round)
            self.best_parameters = parameters
        return float(metrics.get("loss", 0.0)), {"macro_f1": float(metrics["macro_f1"])}

    def finalize(self) -> dict[str, Any]:
        selected = self.best_parameters or self.latest_parameters
        if selected is None:
            raise RuntimeError("No parameters available for P8-b QGA L2 finalization")
        model_cfg = dict(self.config["model"])
        model_cfg["input_dim"] = int(self.mask.sum())
        model = build_model(model_cfg, output_dim=8).to(self.device)
        set_parameters(model, parameters_to_ndarrays(selected))
        torch.save(model.state_dict(), self.checkpoints_dir / "best_global_model.pth")
        torch.save(model.state_dict(), self.checkpoints_dir / "last_global_model.pth")
        val_arrays = concatenate_masked_validation_arrays(
            self.config,
            self.repo_root,
            alpha=self.alpha,
            clients=self.clients,
            mask=self.mask,
            max_samples_per_client=int(self.config["execution"]["smoke_max_samples_per_client"]) if self.mode == "smoke" else None,
        )
        val_result = evaluate_arrays(model=model, arrays=val_arrays, batch_size=2048, device=self.device, seed=int(self.config["training"]["seed"]), class_names=[str(i) for i in range(8)])
        metrics_val = val_result["metrics"]
        metrics_test: dict[str, Any] = {}
        if self.evaluate_test:
            test_arrays = load_masked_global_arrays(
                self.config,
                self.repo_root,
                split="test",
                mask=self.mask,
                max_samples=int(self.config["execution"]["smoke_max_samples_per_client"]) if self.mode == "smoke" else None,
                seed=int(self.config["training"]["seed"]) + 999,
            )
            test_result = evaluate_arrays(model=model, arrays=test_arrays, batch_size=2048, device=self.device, seed=int(self.config["training"]["seed"]), class_names=[str(i) for i in range(8)])
            metrics_test = test_result["metrics"]
            write_json(self.artifacts_dir / "metrics_test.json", metrics_test)
            write_json(self.artifacts_dir / "classification_report.json", classification_report(metrics_test))
            write_rows(self.artifacts_dir / "one_vs_rest_metrics.csv", one_vs_rest_rows(metrics_test))
            _write_csv(self.artifacts_dir / "confusion_matrix.csv", [{"row": idx, **{f"c{j}": value for j, value in enumerate(row)}} for idx, row in enumerate(metrics_test["confusion_matrix"])])
        _write_csv(self.artifacts_dir / "metrics_rounds.csv", self.round_rows)
        _write_csv(self.artifacts_dir / "metrics_clients.csv", self.client_rows)
        _write_csv(self.artifacts_dir / "bandwidth_rounds.csv", self.round_rows)
        _write_csv(self.artifacts_dir / "aggregation_weights.csv", [{"round": row["round"], "weighting": "num_examples"} for row in self.round_rows])
        write_json(self.artifacts_dir / "selected_features_reference.json", self.mask_info["payload"])
        write_json(self.artifacts_dir / "model_config.json", {**model_cfg, "selected_features_count": int(self.mask.sum()), "model_size_bytes": model_size_bytes(model.state_dict())})
        write_json(self.artifacts_dir / "metrics_val.json", metrics_val)
        comparison = {"p6_l2_available": False}
        write_json(self.artifacts_dir / "comparison_with_p6_l2.json", comparison)
        artifacts = [rel(path, self.repo_root) for path in [self.artifacts_dir / name for name in ["metrics_rounds.csv", "metrics_clients.csv", "bandwidth_rounds.csv", "metrics_val.json", "selected_features_reference.json", "model_config.json"]] if path.exists()]
        summary = {
            "accepted": True,
            "phase": "P8-b",
            "task": "l2_family",
            "method": "FedAvg + QGA L2 Flower",
            "mode": self.mode,
            "runtime": "manual_or_subprocess",
            "run_id": self.run_id,
            "true_flower_runtime": True,
            "selected_mask_id": self.mask_info["payload"].get("selected_mask_id") or self.mask_info["payload"].get("mask_id"),
            "selected_mask_source": self.mask_info["payload"].get("selected_mask_source"),
            "calibration_decision_used": bool(self.mask_info["payload"].get("calibration_decision_used")),
            "scenario": {"alpha": self.alpha, "clients": self.clients, "rounds": self.rounds},
            "dataset": {"input_dim_selected": int(self.mask.sum()), "test_sent_to_clients": False, "test_used_for_selection": False},
            "qga": {"selected_features_count": int(self.mask.sum()), "selected_features": self.mask_info["payload"].get("selected_features", [])},
            "training": {"framework": "Flower", "strategy": "FedAvg", "rounds_completed": len(self.round_rows), "best_round": self.best_round},
            "validation": {"metrics": metrics_val},
            "test": {"metrics": metrics_test, "evaluated": self.evaluate_test, "used_for_selection": False},
            "comparison_with_p6_l2": comparison,
            "artifacts": artifacts,
            "figures": [],
            "criteria": {
                "true_flower_runtime": True,
                "p3_l2_partitions_used": True,
                "global_test_holdout_protected": True,
                "test_sent_to_clients_false": True,
                "test_not_used_for_selection": True,
                "final_selected_mask_used": self.mask_info["payload"].get("selected_mask_source") == "final_selected_mask",
                "calibration_decision_used": bool(self.mask_info["payload"].get("calibration_decision_used")),
                "metrics_generated": True,
            },
            "warnings": ["Smoke metrics have low scientific significance."] if self.mode == "smoke" else [],
            "errors": [],
        }
        write_json(self.artifacts_dir / "run_summary.json", summary)
        write_json(self.artifacts_dir / "run_manifest.json", {"run_id": self.run_id, "test_sent_to_clients": False, "run_summary": rel(self.artifacts_dir / "run_summary.json", self.repo_root)})
        write_json(self.run_dir / "manifest.json", {"run_id": self.run_id, "test_sent_to_clients": False})
        write_json(Path(scenario_dir(self.config, alpha=self.alpha, clients=self.clients)) / "latest_run_summary.json", summary)
        return summary


def start_server(config: dict[str, Any], *, alpha: float, clients: int, rounds: int, address: str, mode: str, max_samples_per_client: int | None, run_id: str | None = None, mask_path: str | Path | None = None, evaluate_test: bool = True) -> dict[str, Any]:
    assert_port_available(address)
    mask_info = _load_mask(config, mask_path)
    paths = prepare_run_paths(config, alpha=alpha, clients=clients, run_id=run_id)
    strategy = QGAL2FedAvgStrategy(config=config, alpha=alpha, clients=clients, rounds=rounds, mask_info=mask_info, run_paths=paths, max_samples_per_client=max_samples_per_client, mode=mode, evaluate_test=evaluate_test)
    fl.server.start_server(server_address=address, config=ServerConfig(num_rounds=int(rounds)), strategy=strategy)
    return strategy.finalize()


def start_client(config: dict[str, Any], *, client_id: str, alpha: float, clients: int, address: str, mode: str, max_samples_per_client: int | None, run_id: str | None = None, mask_path: str | Path | None = None) -> None:
    mask_info = _load_mask(config, mask_path)
    resolved_run_id = run_id or latest_run_id(config, alpha=alpha, clients=clients)
    client = QGAL2FlowerClient(config=config, client_id=client_id, alpha=alpha, clients=clients, mask=mask_info["mask"], run_id=resolved_run_id, max_samples_per_client=max_samples_per_client if mode == "smoke" else None)
    fl.client.start_client(server_address=address, client=client.to_client())


def run_smoke_subprocess(
    *,
    config_path: Path,
    alpha: float,
    clients: int,
    rounds: int,
    max_samples_per_client: int,
    address: str,
    timeout_sec: int = 600,
    mask_path: str | Path | None = None,
    evaluate_test: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    assert_port_available(address)
    run_id = make_run_id()
    prepare_run_paths(config, alpha=alpha, clients=clients, run_id=run_id)
    scripts = repo_path(config) / "experiments" / "qi-fl-ids-iot-final" / "src" / "scripts"
    common = ["--config", str(config_path), "--alpha", str(alpha), "--clients", str(clients), "--address", address, "--run-id", run_id, "--mode", "smoke", "--max-samples-per-client", str(max_samples_per_client)]
    if mask_path is not None:
        common.extend(["--mask-path", str(mask_path)])
    server_cmd = [sys.executable, str(scripts / "08_b_start_qga_l2_flower_server.py"), *common, "--rounds", str(rounds)]
    if not evaluate_test:
        server_cmd.append("--validation-only")
    processes: list[tuple[str, subprocess.Popen]] = []
    logs_dir = Path(prepare_run_paths(config, alpha=alpha, clients=clients, run_id=run_id, mark_latest=False)["logs_dir"])
    handles = []
    try:
        server_log = (logs_dir / "server_stdout.log").open("w", encoding="utf-8")
        handles.append(server_log)
        processes.append(("server", subprocess.Popen(server_cmd, cwd=repo_path(config), stdout=server_log, stderr=subprocess.STDOUT, text=True)))
        import time

        time.sleep(4)
        for index in range(1, clients + 1):
            log = (logs_dir / f"client_{index}_stdout.log").open("w", encoding="utf-8")
            handles.append(log)
            cmd = [sys.executable, str(scripts / "08_b_start_qga_l2_flower_client.py"), *common, "--client-id", f"client_{index}"]
            processes.append((f"client_{index}", subprocess.Popen(cmd, cwd=repo_path(config), stdout=log, stderr=subprocess.STDOUT, text=True)))
        deadline = time.time() + int(timeout_sec)
        failures = []
        for name, proc in processes:
            remaining = max(1, int(deadline - time.time()))
            try:
                code = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
                failures.append((name, "timeout"))
                continue
            if code != 0:
                failures.append((name, code))
        if failures:
            raise RuntimeError(f"P8-b QGA L2 Flower subprocess failure(s): {failures}")
    finally:
        for _, proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for handle in handles:
            handle.close()
    summary_path = scenario_dir(config, alpha=alpha, clients=clients) / "runs" / run_id / "artifacts" / "run_summary.json"
    return load_json(summary_path)
