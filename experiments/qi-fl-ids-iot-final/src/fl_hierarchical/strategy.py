"""True Flower FedAvg strategy for P6 hierarchical L2/L3 tasks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import flwr as fl
import numpy as np
import torch
from flwr.common import FitIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from torch import nn

from fl_l1.round_logger import ConsoleLogger
from fl_l1.scenario_loader import rel
from fl_hierarchical.communication import model_size_bytes, round_bandwidth
from fl_hierarchical.data import (
    HierarchicalArrays,
    L2IndexScenario,
    TaskSpec,
    load_global_arrays,
    make_dataloader,
    write_json,
)
from fl_hierarchical.metrics import (
    classification_report,
    multiclass_metrics,
    one_vs_rest_rows,
    top_confusion_pairs,
    write_rows,
)
from fl_hierarchical.models import build_model, get_parameters, set_parameters
from fl_hierarchical.plotting import generate_hierarchical_figures
from fl_hierarchical.summary_schema import (
    architecture_string,
    existing_relative_paths,
    figure_dir,
    run_artifact_paths,
    run_criteria,
    run_figure_paths,
    write_latest_run_summary,
)


def select_device(device_config: str) -> torch.device:
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def _append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_confusion_csv(path: Path, matrix: list[list[int]], class_names: list[str]) -> None:
    rows = []
    for index, row in enumerate(matrix):
        payload = {"true_class": class_names[index]}
        payload.update({f"pred_{name}": int(value) for name, value in zip(class_names, row)})
        rows.append(payload)
    _write_csv(path, rows)


def _mean_per_class(metrics: dict[str, Any], key: str) -> float:
    values = [float(payload[key]) for payload in metrics.get("per_class", {}).values()]
    return float(np.mean(values)) if values else 0.0


def train_local(
    *,
    model: nn.Module,
    arrays: HierarchicalArrays,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    num_classes: int,
    use_class_weights: bool,
) -> dict[str, Any]:
    weights = None
    if use_class_weights:
        counts = np.bincount(arrays.y.astype(np.int64), minlength=int(num_classes)).astype(np.float32)
        weights_np = np.zeros_like(counts, dtype=np.float32)
        non_zero = counts > 0
        weights_np[non_zero] = counts.sum() / (float(num_classes) * counts[non_zero])
        weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    loader = make_dataloader(arrays, batch_size=batch_size, shuffle=True, seed=seed, device=device)
    start = perf_counter()
    last_loss = 0.0
    for _ in range(int(local_epochs)):
        model.train()
        running_loss = 0.0
        batches = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1
        last_loss = running_loss / max(batches, 1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"loss": float(last_loss), "fit_time_sec": float(perf_counter() - start)}


def evaluate_arrays(
    *,
    model: nn.Module,
    arrays: HierarchicalArrays,
    batch_size: int,
    device: torch.device,
    seed: int,
    class_names: list[str],
) -> dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    loader = make_dataloader(arrays, batch_size=batch_size, shuffle=False, seed=seed, device=device)
    start = perf_counter()
    losses: list[float] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            losses.append(float(loss.item()))
            y_true.append(y_batch.detach().cpu().numpy())
            y_pred.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    true = np.concatenate(y_true) if y_true else np.asarray([], dtype=np.int64)
    pred = np.concatenate(y_pred) if y_pred else np.asarray([], dtype=np.int64)
    metrics = multiclass_metrics(true, pred, class_names)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["FPR_macro"] = _mean_per_class(metrics, "FPR")
    metrics["FNR_macro"] = _mean_per_class(metrics, "FNR")
    metrics["class_names"] = class_names
    return {
        "metrics": metrics,
        "y_true": true,
        "y_pred": pred,
        "eval_time_sec": float(perf_counter() - start),
    }


def client_metrics_payload(
    *,
    client_id: str,
    round_number: int,
    train_arrays: HierarchicalArrays,
    val_arrays: HierarchicalArrays,
    train_loss: float,
    val_result: dict[str, Any],
    fit_time_sec: float,
    payload_size: int,
) -> dict[str, Any]:
    metrics = val_result["metrics"]
    return {
        "round": int(round_number),
        "client_id": client_id,
        "train_samples": int(train_arrays.num_samples),
        "val_samples": int(val_arrays.num_samples),
        "local_train_loss": float(train_loss),
        "local_val_loss": float(metrics["loss"]),
        "local_accuracy": float(metrics["accuracy"]),
        "local_macro_f1": float(metrics["macro_f1"]),
        "local_weighted_f1": float(metrics["weighted_f1"]),
        "local_precision_macro": float(metrics["precision_macro"]),
        "local_recall_macro": float(metrics["recall_macro"]),
        "local_fpr_macro": float(metrics["FPR_macro"]),
        "local_fnr_macro": float(metrics["FNR_macro"]),
        "fit_time_sec": float(fit_time_sec),
        "eval_time_sec": float(val_result["eval_time_sec"]),
        "upload_bytes": int(payload_size),
        "download_bytes": int(payload_size),
    }


class HierarchicalFedAvgStrategy(FedAvg):
    """Flower FedAvg with rich P6 artifacts."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        repo_root: Path,
        scenario: L2IndexScenario,
        task_spec: TaskSpec,
        run_dir: Path,
        validation_arrays: HierarchicalArrays,
        max_samples_per_client: int | None,
        mode: str,
        run_id: str | None = None,
        runtime_mode: str = "legacy-local",
    ) -> None:
        self.config = config
        self.repo_root = repo_root
        self.scenario = scenario
        self.task_spec = task_spec
        self.run_dir = run_dir
        self.run_id = run_id or run_dir.name
        self.validation_arrays = validation_arrays
        self.max_samples_per_client = max_samples_per_client
        self.mode = mode
        self.runtime_mode = runtime_mode
        self.training_cfg = config["training"]
        self.device = select_device(str(self.training_cfg["device"]))
        self.checkpoints_dir = run_dir / "checkpoints"
        self.artifacts_dir = run_dir / "artifacts"
        self.logs_dir = run_dir / "logs"
        for directory in [self.checkpoints_dir, self.artifacts_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        self.console = ConsoleLogger(self.logs_dir / "run_console.log", reset=True)
        (self.logs_dir / "flower_server.log").write_text("", encoding="utf-8")
        (self.logs_dir / "flower_clients.log").touch()
        for path in [
            self.artifacts_dir / "metrics_rounds.csv",
            self.artifacts_dir / "metrics_clients.csv",
            self.artifacts_dir / "bandwidth_rounds.csv",
            self.artifacts_dir / "aggregation_weights.csv",
        ]:
            if path.exists():
                path.unlink()
        self.client_rows: list[dict[str, Any]] = []
        self.round_rows: list[dict[str, Any]] = []
        self.cumulative_bytes = 0
        self.best_macro_f1 = -1.0
        self.best_round = 0
        self.best_parameters: Parameters | None = None
        self.latest_parameters: Parameters | None = None
        self.round_start: dict[int, float] = {}
        model = build_model(task_spec.model_config, output_dim=task_spec.output_dim)
        initial_parameters = ndarrays_to_parameters(get_parameters(model))
        flower_cfg = config["flower"]
        super().__init__(
            fraction_fit=float(flower_cfg["fraction_fit"]),
            fraction_evaluate=float(flower_cfg["fraction_evaluate"]),
            min_fit_clients=int(flower_cfg["min_fit_clients"]),
            min_evaluate_clients=int(flower_cfg["min_evaluate_clients"]),
            min_available_clients=int(flower_cfg["min_available_clients"]),
            initial_parameters=initial_parameters,
        )
        self._server_log(
            f"Starting true Flower P6 server | task={task_spec.task} alpha={scenario.alpha} K={scenario.num_clients} mode={mode}"
        )

    def _server_log(self, message: str) -> None:
        line = f"Flower server | {message}"
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

    def _evaluate_validation(self, parameters: Parameters) -> dict[str, Any]:
        model = build_model(self.task_spec.model_config, output_dim=self.task_spec.output_dim).to(self.device)
        set_parameters(model, parameters_to_ndarrays(parameters))
        return evaluate_arrays(
            model=model,
            arrays=self.validation_arrays,
            batch_size=int(self.training_cfg["batch_size"]) * 4,
            device=self.device,
            seed=int(self.training_cfg["seed"]),
            class_names=self.task_spec.class_names,
        )

    def aggregate_fit(self, server_round: int, results, failures):
        aggregation_start = perf_counter()
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        aggregation_time_sec = perf_counter() - aggregation_start
        if aggregated_parameters is None:
            self._server_log(f"Round {server_round} aggregation returned no parameters")
            return aggregated_parameters, aggregated_metrics
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        train_losses = []
        train_weights = []
        for _, fit_res in results:
            metrics = dict(fit_res.metrics or {})
            self.client_rows.append(metrics)
            train_losses.append(float(metrics.get("local_train_loss", 0.0)))
            train_weights.append(int(fit_res.num_examples))
            _append_csv(
                self.artifacts_dir / "metrics_clients.csv",
                metrics,
                [
                    "round",
                    "client_id",
                    "train_samples",
                    "val_samples",
                    "local_train_loss",
                    "local_val_loss",
                    "local_accuracy",
                    "local_macro_f1",
                    "local_weighted_f1",
                    "local_precision_macro",
                    "local_recall_macro",
                    "local_fpr_macro",
                    "local_fnr_macro",
                    "fit_time_sec",
                    "eval_time_sec",
                    "upload_bytes",
                    "download_bytes",
                ],
            )
            weight = float(fit_res.num_examples / total_examples) if total_examples else 0.0
            _append_csv(
                self.artifacts_dir / "aggregation_weights.csv",
                {
                    "round": int(server_round),
                    "client_id": metrics.get("client_id", "unknown"),
                    "num_examples": int(fit_res.num_examples),
                    "aggregation_weight": weight,
                },
                ["round", "client_id", "num_examples", "aggregation_weight"],
            )
            self._server_log(
                f"Client fit completed | round={server_round} client={metrics.get('client_id', 'unknown')} samples={fit_res.num_examples}"
            )
        self._server_log("Aggregating updates with Flower FedAvg weighted by num_examples")
        val_result = self._evaluate_validation(aggregated_parameters)
        metrics = val_result["metrics"]
        size_bytes = model_size_bytes(parameters_to_ndarrays(aggregated_parameters))
        bandwidth = round_bandwidth(
            model_size_bytes_value=size_bytes,
            num_clients=len(results),
            previous_cumulative_bytes=self.cumulative_bytes,
        )
        self.cumulative_bytes = int(bandwidth["cumulative_bytes"])
        round_time = perf_counter() - self.round_start.pop(int(server_round), perf_counter())
        row = {
            "round": int(server_round),
            "alpha": float(self.scenario.alpha),
            "num_clients": int(self.scenario.num_clients),
            "train_loss_mean": float(np.average(train_losses, weights=train_weights)) if train_weights else 0.0,
            "val_loss_mean": float(metrics["loss"]),
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
            "precision_macro": float(metrics["precision_macro"]),
            "recall_macro": float(metrics["recall_macro"]),
            "FPR_macro": float(metrics["FPR_macro"]),
            "FNR_macro": float(metrics["FNR_macro"]),
            "round_time_sec": float(round_time),
            "aggregation_time_sec": float(aggregation_time_sec),
            "model_size_bytes": int(size_bytes),
            "communication_upload_bytes": int(bandwidth["upload_bytes"]),
            "communication_download_bytes": int(bandwidth["download_bytes"]),
            "communication_total_bytes": int(bandwidth["total_bytes"]),
            "communication_cumulative_bytes": int(bandwidth["cumulative_bytes"]),
        }
        self.round_rows.append(row)
        _append_csv(
            self.artifacts_dir / "metrics_rounds.csv",
            row,
            list(row.keys()),
        )
        _append_csv(self.artifacts_dir / "bandwidth_rounds.csv", {"round": int(server_round), **bandwidth}, ["round", "upload_bytes", "download_bytes", "total_bytes", "cumulative_bytes", "total_mb", "cumulative_mb"])
        self.console.log(
            f"[Round {server_round:02d}/{int(self.config['scenario']['rounds']):02d}] "
            f"task={self.task_spec.short_name} loss={row['train_loss_mean']:.4f} "
            f"val_loss={row['val_loss_mean']:.4f} macro_f1={row['macro_f1']:.4f} "
            f"FPR_macro={row['FPR_macro']:.4f} bytes={row['communication_total_bytes']} "
            f"cum={row['communication_cumulative_bytes']}"
        )
        self.latest_parameters = aggregated_parameters
        self._save_checkpoint("last_global_model.pth", aggregated_parameters, server_round, row)
        if float(row["macro_f1"]) > self.best_macro_f1:
            self.best_macro_f1 = float(row["macro_f1"])
            self.best_round = int(server_round)
            self.best_parameters = aggregated_parameters
            self._save_checkpoint("best_global_model.pth", aggregated_parameters, server_round, row)
            self._server_log(f"Saving best checkpoint if improved | improved=true round={server_round}")
        return aggregated_parameters, aggregated_metrics

    def _save_checkpoint(self, filename: str, parameters: Parameters, server_round: int, metrics: dict[str, Any]) -> None:
        model = build_model(self.task_spec.model_config, output_dim=self.task_spec.output_dim)
        set_parameters(model, parameters_to_ndarrays(parameters))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "task": self.task_spec.task,
                "round": int(server_round),
                "selection_metric": "server_validation_macro_f1",
                "selection_metric_value": float(metrics["macro_f1"]),
                "selection_split": "server_validation",
                "test_used_for_selection": False,
                "flower_runtime": True,
            },
            self.checkpoints_dir / filename,
        )

    def finalize(self) -> dict[str, Any]:
        selected = self.best_parameters or self.latest_parameters
        if selected is None:
            raise RuntimeError("No Flower parameters available to finalize P6 run")
        model = build_model(self.task_spec.model_config, output_dim=self.task_spec.output_dim).to(self.device)
        set_parameters(model, parameters_to_ndarrays(selected))
        val_result = evaluate_arrays(
            model=model,
            arrays=self.validation_arrays,
            batch_size=int(self.training_cfg["batch_size"]) * 4,
            device=self.device,
            seed=int(self.training_cfg["seed"]),
            class_names=self.task_spec.class_names,
        )
        test_arrays = load_global_arrays(
            self.config,
            self.repo_root,
            split="test",
            task_spec=self.task_spec,
            max_samples=self.max_samples_per_client if self.mode == "smoke" else None,
            seed=int(self.training_cfg["seed"]) + 90_000,
        )
        test_result = evaluate_arrays(
            model=model,
            arrays=test_arrays,
            batch_size=int(self.training_cfg["batch_size"]) * 4,
            device=self.device,
            seed=int(self.training_cfg["seed"]) + 91_000,
            class_names=self.task_spec.class_names,
        )
        metrics_val = val_result["metrics"]
        metrics_test = test_result["metrics"]
        metrics_test["top_confusion_pairs"] = top_confusion_pairs(metrics_test["confusion_matrix"], self.task_spec.class_names)
        num_parameters = int(model.count_parameters())
        model_size = int(model_size_bytes(model.state_dict()))
        model_config = {
            "name": self.task_spec.model_config["name"],
            "architecture": architecture_string(self.task_spec.model_config),
            "config": self.task_spec.model_config,
            "num_parameters": num_parameters,
            "model_size_bytes": model_size,
            "checkpoint_selection_metric": "val_macro_f1",
            "checkpoint_selection_split": "server_validation",
        }
        write_json(self.artifacts_dir / "model_config.json", model_config)
        write_json(self.artifacts_dir / "class_mapping.json", self.task_spec.class_mapping)
        write_json(self.artifacts_dir / "metrics_val.json", metrics_val)
        write_json(self.artifacts_dir / "metrics_test.json", metrics_test)
        write_json(self.artifacts_dir / "classification_report.json", classification_report(metrics_test))
        write_json(
            self.artifacts_dir / "threshold_or_decision_config.json",
            {
                "decision_rule": "argmax_multiclass_logits",
                "threshold_required": False,
                "selection_split": "server_validation",
                "test_used_for_selection": False,
            },
        )
        _write_confusion_csv(self.artifacts_dir / "confusion_matrix.csv", metrics_test["confusion_matrix"], self.task_spec.class_names)
        one_vs_rest = one_vs_rest_rows(metrics_test)
        write_rows(self.artifacts_dir / "one_vs_rest_metrics.csv", one_vs_rest)
        p4_path = self.repo_root / self.config["inputs"]["p4_l1_metrics"]
        p4_metrics = json.loads(p4_path.read_text(encoding="utf-8")) if p4_path.exists() else {}
        comparison = {
            "p4_l1_accuracy": float(p4_metrics.get("accuracy", 0.0)),
            "p4_l1_macro_f1": float(p4_metrics.get("macro_f1", 0.0)),
            "p6_task": self.task_spec.task,
            "p6_accuracy": float(metrics_test["accuracy"]),
            "p6_macro_f1": float(metrics_test["macro_f1"]),
            "note": "L1 is production; P6 L2/L3 are experimental and not directly comparable to the binary L1 task.",
        }
        write_json(self.artifacts_dir / "comparison_with_l1_l2_l3.json", comparison)
        figures_dir = figure_dir(
            self.config,
            self.repo_root,
            task=self.task_spec.task,
            alpha=float(self.scenario.alpha),
            clients=int(self.scenario.num_clients),
            run_id=self.run_id,
        )
        generate_hierarchical_figures(
            figures_dir=figures_dir,
            task=self.task_spec.task,
            round_rows=self.round_rows,
            client_rows=self.client_rows,
            metrics_test=metrics_test,
            comparison=comparison,
            model_cfg=self.task_spec.model_config,
            num_parameters=num_parameters,
        )
        client_train_rows = {client.client_id: int(client.train_samples) for client in self.scenario.clients}
        client_val_rows = {client.client_id: int(client.val_samples) for client in self.scenario.clients}
        client_train_rows_used = {
            client.client_id: int(min(client.train_samples, self.max_samples_per_client))
            if self.max_samples_per_client is not None
            else int(client.train_samples)
            for client in self.scenario.clients
        }
        client_val_rows_used = {
            client.client_id: int(min(client.val_samples, self.max_samples_per_client))
            if self.max_samples_per_client is not None
            else int(client.val_samples)
            for client in self.scenario.clients
        }
        rounds_completed = len(self.round_rows)
        rounds_configured = int(self.config["scenario"]["rounds"])
        docs_path = self.repo_root / self.config["final_experiment_dir"] / "docs" / "06_hierarchical_l2_l3.md"
        artifacts = existing_relative_paths(run_artifact_paths(self.run_dir), self.repo_root)
        figures = existing_relative_paths(run_figure_paths(figures_dir, self.task_spec.task), self.repo_root)
        criteria = run_criteria(
            task=self.task_spec.task,
            mode=self.mode,
            rounds_completed=rounds_completed,
            rounds_configured=rounds_configured,
            artifacts=artifacts,
            figures=figures,
            docs_generated=docs_path.exists(),
        )
        warnings: list[str] = []
        if self.mode == "smoke":
            warnings.append("Smoke run uses sampled clients/test data; metrics have low scientific significance.")
        accepted = all(
            bool(criteria[key])
            for key in [
                "true_flower_runtime",
                "p3_l2_partitions_used",
                "global_test_holdout_protected",
                "test_sent_to_clients_false",
                "metrics_test_generated",
                "one_vs_rest_metrics_generated",
                "figures_generated",
                "l2_l3_not_deployed",
            ]
        )
        summary = {
            "accepted": accepted,
            "phase": "P6",
            "task": self.task_spec.task,
            "mode": self.mode,
            "runtime": self.runtime_mode,
            "run_id": self.run_id,
            "scenario": {
                "alpha": float(self.scenario.alpha),
                "clients": int(self.scenario.num_clients),
                "client_ids": [client.client_id for client in self.scenario.clients],
                "rounds": rounds_configured,
            },
            "dataset": {
                "input_dim": int(self.task_spec.model_config["input_dim"]),
                "train_rows_total": int(sum(client_train_rows.values())),
                "val_rows_total": int(sum(client_val_rows.values())),
                "test_rows": int(metrics_test["support_total"]),
                "client_train_rows": client_train_rows,
                "client_val_rows": client_val_rows,
                "client_train_rows_used": client_train_rows_used,
                "client_val_rows_used": client_val_rows_used,
                "global_test_holdout": rel(self.scenario.global_test_npz, self.repo_root),
                "test_sent_to_clients": False,
            },
            "model": model_config,
            "training": {
                "framework": "Flower",
                "flower_version": fl.__version__,
                "strategy": "FedAvg",
                "local_epochs": int(self.training_cfg["local_epochs"]),
                "rounds_configured": rounds_configured,
                "rounds_completed": rounds_completed,
                "best_round": int(self.best_round),
                "selection_metric": "val_macro_f1",
                "selection_split": "server_validation",
            },
            "validation": {
                "metrics": metrics_val,
                "rows": int(metrics_val["support_total"]),
                "scientific_significance": "low_for_smoke" if self.mode == "smoke" else "valid_for_run",
            },
            "test": {
                "metrics": metrics_test,
                "rows": int(metrics_test["support_total"]),
                "global_holdout": rel(self.scenario.global_test_npz, self.repo_root),
                "scientific_significance": "low_for_smoke" if self.mode == "smoke" else "valid_for_run",
            },
            "one_vs_rest": {
                "csv": rel(self.artifacts_dir / "one_vs_rest_metrics.csv", self.repo_root),
                "summary": {
                    "classes": len(one_vs_rest),
                    "mean_f1": float(np.mean([row["f1"] for row in one_vs_rest])) if one_vs_rest else 0.0,
                },
            },
            "comparison_with_l1_l2_l3": comparison,
            "artifacts": artifacts,
            "figures": figures,
            "criteria": criteria,
            "warnings": warnings,
            "errors": [],
            "scientific_significance": "low_for_smoke" if self.mode == "smoke" else "valid_for_run",
            "round_rows": self.round_rows,
        }
        run_summary_path = self.artifacts_dir / "run_summary.json"
        run_manifest_path = self.artifacts_dir / "run_manifest.json"
        root_manifest_path = self.run_dir / "manifest.json"
        manifest = {
            "run_id": self.run_id,
            "accepted": accepted,
            "phase": "P6",
            "task": self.task_spec.task,
            "flower_runtime": True,
            "run_summary": rel(run_summary_path, self.repo_root),
            "scenario_manifest": rel(self.scenario.manifest_path, self.repo_root),
            "global_test_reference": rel(self.scenario.global_test_reference_path, self.repo_root),
            "global_test_holdout": rel(self.scenario.global_test_npz, self.repo_root),
            "test_sent_to_clients": False,
            "checkpoints_dir": rel(self.checkpoints_dir, self.repo_root),
            "artifacts_dir": rel(self.artifacts_dir, self.repo_root),
            "logs_dir": rel(self.logs_dir, self.repo_root),
            "figures_dir": rel(figures_dir, self.repo_root),
            "criteria": criteria,
        }
        write_json(run_manifest_path, manifest)
        write_json(root_manifest_path, manifest)
        write_json(run_summary_path, summary)
        artifacts = existing_relative_paths(run_artifact_paths(self.run_dir), self.repo_root)
        figures = existing_relative_paths(run_figure_paths(figures_dir, self.task_spec.task), self.repo_root)
        summary["artifacts"] = artifacts
        summary["figures"] = figures
        summary["criteria"] = run_criteria(
            task=self.task_spec.task,
            mode=self.mode,
            rounds_completed=rounds_completed,
            rounds_configured=rounds_configured,
            artifacts=artifacts,
            figures=figures,
            docs_generated=docs_path.exists(),
        )
        summary["accepted"] = all(
            bool(summary["criteria"][key])
            for key in [
                "true_flower_runtime",
                "p3_l2_partitions_used",
                "global_test_holdout_protected",
                "test_sent_to_clients_false",
                "metrics_test_generated",
                "one_vs_rest_metrics_generated",
                "figures_generated",
                "l2_l3_not_deployed",
            ]
        )
        write_json(run_summary_path, summary)
        latest_summary_path = write_latest_run_summary(run_dir=self.run_dir, repo_root=self.repo_root, summary=summary)
        summary["latest_run_summary"] = rel(latest_summary_path, self.repo_root)
        write_json(run_summary_path, summary)
        write_json(latest_summary_path, summary)
        self._server_log(
            f"Final evaluation on global test holdout | task={self.task_spec.task} macro_f1={metrics_test['macro_f1']:.4f}"
        )
        return summary


def build_initial_parameters(config: dict[str, Any], task_spec: TaskSpec) -> Parameters:
    return ndarrays_to_parameters(get_parameters(build_model(task_spec.model_config, output_dim=task_spec.output_dim)))
