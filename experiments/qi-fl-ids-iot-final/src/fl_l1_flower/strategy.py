"""Flower FedAvg strategy with P5-compatible logging and artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import flwr as fl
from flwr.common import FitIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fl_l1.client_data import load_client_npz
from fl_l1.evaluation import finalize_test_metrics, tune_threshold_on_validation
from fl_l1.round_logger import ConsoleLogger, RoundLogger, format_round_console_line
from fl_l1.scenario_loader import rel, write_json
from fl_l1_flower.communication import model_size_bytes, round_bandwidth
from fl_l1_flower.metrics import aggregate_evaluate_metrics, aggregate_fit_metrics, client_metrics_row
from fl_l1_flower.plotting import generate_flower_figures
from fl_l1_flower.summary_schema import (
    architecture_string,
    comparison_with_p4,
    existing_relative_paths,
    figure_dir,
    run_artifact_paths,
    run_criteria,
    run_figure_paths,
    write_latest_run_summary,
)
from fl_l1_flower.task import build_model, evaluate_arrays, get_parameters, select_device, set_parameters


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
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


class FlowerL1FedAvgStrategy(FedAvg):
    """FedAvg strategy executed by Flower, with final P5 artifacts."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        repo_root: Path,
        scenario,
        run_dir: Path,
        validation_arrays,
        max_samples_per_client: int | None,
        mode: str,
        run_id: str | None = None,
        runtime_mode: str = "legacy-local",
    ) -> None:
        self.config = config
        self.repo_root = repo_root
        self.scenario = scenario
        self.run_dir = run_dir
        self.run_id = run_id or run_dir.name
        self.validation_arrays = validation_arrays
        self.max_samples_per_client = max_samples_per_client
        self.mode = mode
        self.runtime_mode = runtime_mode
        self.training_cfg = config["training"]
        self.threshold_cfg = config["threshold"]
        self.flower_cfg = config["flower"]
        self.device = select_device(str(self.training_cfg["device"]))
        self.checkpoints_dir = run_dir / "checkpoints"
        self.artifacts_dir = run_dir / "artifacts"
        self.logs_dir = run_dir / "logs"
        for directory in [self.checkpoints_dir, self.artifacts_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        self.round_logger = RoundLogger(self.artifacts_dir, self.logs_dir, reset=True)
        self.console = ConsoleLogger(self.logs_dir / "run_console.log", reset=True)
        (self.logs_dir / "flower_server.log").write_text("", encoding="utf-8")
        self.client_rows: list[dict[str, Any]] = []
        self.round_rows: list[dict[str, Any]] = []
        self.threshold_rows: list[dict[str, Any]] = []
        self.cumulative_bytes = 0
        self.best_macro_f1 = -1.0
        self.best_round = 0
        self.best_parameters: Parameters | None = None
        self.latest_parameters: Parameters | None = None
        self.round_start: dict[int, float] = {}
        initial_model = build_model(config["model"])
        initial_parameters = ndarrays_to_parameters(get_parameters(initial_model))

        super().__init__(
            fraction_fit=float(self.flower_cfg["fraction_fit"]),
            fraction_evaluate=float(self.flower_cfg["fraction_evaluate"]),
            min_fit_clients=int(self.flower_cfg["min_fit_clients"]),
            min_evaluate_clients=int(self.flower_cfg["min_evaluate_clients"]),
            min_available_clients=int(self.flower_cfg["min_available_clients"]),
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        )
        self._server_log(
            f"Starting true Flower FedAvg L1 server | alpha={scenario.alpha} K={scenario.num_clients} mode={mode}"
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
        updated = []
        for client_proxy, fit_ins in configured:
            cfg = dict(fit_ins.config)
            cfg["server_round"] = int(server_round)
            updated.append((client_proxy, FitIns(parameters=fit_ins.parameters, config=cfg)))
        return updated

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager):
        configured = super().configure_evaluate(server_round, parameters, client_manager)
        updated = []
        for client_proxy, evaluate_ins in configured:
            cfg = dict(evaluate_ins.config)
            cfg["server_round"] = int(server_round)
            updated.append((client_proxy, type(evaluate_ins)(parameters=evaluate_ins.parameters, config=cfg)))
        return updated

    def _evaluate_server_validation(self, parameters: Parameters, *, threshold: float) -> dict[str, Any]:
        model = build_model(self.config["model"]).to(self.device)
        set_parameters(model, parameters_to_ndarrays(parameters))
        return evaluate_arrays(
            model=model,
            arrays=self.validation_arrays,
            batch_size=int(self.training_cfg["batch_size"]) * 4,
            device=self.device,
            seed=int(self.training_cfg["seed"]),
            threshold=threshold,
            collect_probabilities=True,
        )

    def aggregate_fit(self, server_round: int, results, failures):
        aggregation_start = perf_counter()
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        aggregation_time_sec = perf_counter() - aggregation_start
        if aggregated_parameters is None:
            self._server_log(f"Round {server_round} aggregation returned no parameters")
            return aggregated_parameters, aggregated_metrics

        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        train_loss_values: list[float] = []
        train_weights: list[int] = []
        for client_proxy, fit_res in results:
            metrics = dict(fit_res.metrics or {})
            row = client_metrics_row(metrics)
            self.round_logger.log_client(row)
            self.client_rows.append(row)
            train_loss_values.append(float(row["local_train_loss"]))
            train_weights.append(int(fit_res.num_examples))
            weight = float(fit_res.num_examples / total_examples) if total_examples else 0.0
            self.round_logger.log_aggregation_weight(
                {
                    "round": int(server_round),
                    "client_id": row["client_id"],
                    "num_examples": int(fit_res.num_examples),
                    "aggregation_weight": weight,
                }
            )
            self._server_log(
                f"Client fit completed | round={server_round} client={row['client_id']} samples={fit_res.num_examples}"
            )

        self._server_log("Aggregating updates with Flower FedAvg weighted by num_examples")
        train_loss_mean = float(np.average(train_loss_values, weights=train_weights)) if train_weights else 0.0
        self._server_log(f"Evaluating global model on federated validation | round={server_round}")
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
            "train_loss_mean": train_loss_mean,
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
        self.round_logger.log_event({"event": "flower_round_completed", **row})
        self.round_rows.append(row)
        self.console.log(
            format_round_console_line(row, current_round=int(server_round), total_rounds=int(self.config["scenario"]["rounds"]))
        )
        self._server_log(
            f"round aggregated | round={server_round} macro_f1={row['macro_f1']:.4f} "
            f"cumulative_bytes={row['communication_cumulative_bytes']}"
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

    def aggregate_evaluate(self, server_round: int, results, failures):
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        self._server_log(f"Flower client evaluation aggregated | round={server_round}")
        return loss_aggregated, metrics_aggregated

    def _save_checkpoint(self, filename: str, parameters: Parameters, server_round: int, metrics: dict[str, Any]) -> None:
        model = build_model(self.config["model"])
        set_parameters(model, parameters_to_ndarrays(parameters))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
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
        """Write final validation/test artifacts after Flower simulation finishes."""

        selected = self.best_parameters or self.latest_parameters
        if selected is None:
            raise RuntimeError("No Flower parameters available to finalize run")
        model = build_model(self.config["model"]).to(self.device)
        set_parameters(model, parameters_to_ndarrays(selected))
        val_result = evaluate_arrays(
            model=model,
            arrays=self.validation_arrays,
            batch_size=int(self.training_cfg["batch_size"]) * 4,
            device=self.device,
            seed=int(self.training_cfg["seed"]),
            threshold=0.5,
            collect_probabilities=True,
        )
        threshold_payload, threshold_rows = tune_threshold_on_validation(
            val_result["y_true"],
            val_result["prob_attack"],
            start=float(self.threshold_cfg["start"]),
            stop=float(self.threshold_cfg["stop"]),
            step=float(self.threshold_cfg["step"]),
        )
        threshold_payload = dict(threshold_payload)
        threshold_payload["selection_split"] = "server_validation"
        threshold_payload["test_used_for_threshold"] = False
        primary_threshold = float(threshold_payload["primary_threshold"])
        metrics_val = dict(threshold_payload["primary_validation_metrics"])
        metrics_val["selection_split"] = "server_validation"
        metrics_val["test_used_for_threshold"] = False

        test_arrays = load_client_npz(
            self.repo_root / self.config["inputs"]["global_test_npz"],
            max_samples=self.max_samples_per_client if self.mode == "smoke" else None,
            seed=int(self.training_cfg["seed"]) + 99_000,
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
        metrics_test["model_size_bytes"] = model_size_bytes(model.state_dict())
        metrics_test["num_parameters"] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        num_parameters = int(metrics_test["num_parameters"])
        model_size = int(metrics_test["model_size_bytes"])
        model_config = {
            "name": "FlowerL1FedAvgMLP",
            "architecture": architecture_string(self.config["model"]),
            "config": self.config["model"],
            "num_parameters": num_parameters,
            "model_size_bytes": model_size,
            "checkpoint_selection_metric": "val_macro_f1",
            "checkpoint_selection_split": "server_validation",
        }

        write_json(self.artifacts_dir / "model_config.json", model_config)
        write_json(self.artifacts_dir / "metrics_val.json", metrics_val)
        write_json(self.artifacts_dir / "metrics_test.json", metrics_test)
        write_json(self.artifacts_dir / "threshold.json", threshold_payload)
        _write_csv(self.artifacts_dir / "threshold_sweep.csv", threshold_rows)
        _write_confusion_matrix(self.artifacts_dir / "confusion_matrix.csv", metrics_test)
        write_json(self.artifacts_dir / "classification_report.json", metrics_test["classification_report"])
        p4_path = self.repo_root / self.config["inputs"]["centralized_l1_metrics"]
        p4_metrics = json.loads(p4_path.read_text(encoding="utf-8")) if p4_path.exists() else {}
        comparison = comparison_with_p4(p4_metrics, metrics_test)
        write_json(self.artifacts_dir / "comparison_with_p4.json", comparison)

        figures_dir = figure_dir(
            self.config,
            self.repo_root,
            alpha=float(self.scenario.alpha),
            clients=int(self.scenario.num_clients),
            run_id=self.run_id,
        )
        generate_flower_figures(
            figures_dir=figures_dir,
            round_rows=self.round_rows,
            client_rows=self.client_rows,
            threshold_rows=threshold_rows,
            metrics_test=metrics_test,
            comparison=comparison,
            y_true_test=test_result["y_true"],
            prob_attack_test=test_result["prob_attack"],
            model_cfg=self.config["model"],
            num_parameters=num_parameters,
        )

        with np.load(self.scenario.global_test_npz, allow_pickle=False) as npz:
            global_test_rows_expected = int(npz["y_binary"].shape[0] if "y_binary" in npz.files else npz["X"].shape[0])
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

        run_summary_path = self.artifacts_dir / "run_summary.json"
        run_manifest_path = self.artifacts_dir / "run_manifest.json"
        root_manifest_path = self.run_dir / "manifest.json"
        artifact_paths = run_artifact_paths(self.run_dir)
        artifact_paths_with_planned = [
            path
            for path in artifact_paths
            if path.exists() or path in {run_summary_path, run_manifest_path, root_manifest_path}
        ]
        artifacts = [rel(path, self.repo_root) for path in artifact_paths_with_planned]
        figures = existing_relative_paths(run_figure_paths(figures_dir), self.repo_root)
        docs_path = self.repo_root / self.config["final_experiment_dir"] / "docs" / "05_2_flower_runtime.md"
        criteria = run_criteria(
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
        if metrics_test.get("roc_pr_warning"):
            warnings.append(str(metrics_test["roc_pr_warning"]))
        errors: list[str] = []
        accepted = all(
            bool(criteria[key])
            for key in [
                "p3_partitions_used",
                "global_test_holdout_protected",
                "test_sent_to_clients_false",
                "server_client_runtime_started",
                "round_metrics_generated",
                "client_metrics_generated",
                "bandwidth_metrics_generated",
                "best_model_saved",
                "last_model_saved",
                "threshold_generated",
                "threshold_validation_only",
                "metrics_val_generated",
                "metrics_test_generated",
                "comparison_with_p4_generated",
                "figures_generated",
                "run_logs_generated",
                "flower_runtime_true",
            ]
        ) and not errors
        summary = {
            "accepted": accepted,
            "runtime": self.runtime_mode,
            "run_id": self.run_id,
            "mode": self.mode,
            "flower_runtime": True,
            "flower_strategy": "FedAvg",
            "dataset_level": "l1_binary",
            "alpha": float(self.scenario.alpha),
            "num_clients": int(self.scenario.num_clients),
            "rounds": int(self.config["scenario"]["rounds"]),
            "scenario": {
                "alpha": float(self.scenario.alpha),
                "clients": int(self.scenario.num_clients),
                "client_ids": [client.client_id for client in self.scenario.clients],
                "rounds": rounds_configured,
            },
            "dataset": {
                "input_dim": int(self.config["model"]["input_dim"]),
                "labels": {"normal": 0, "attack": 1},
                "train_rows_total": int(sum(client_train_rows.values())),
                "val_rows_total": int(sum(client_val_rows.values())),
                "train_rows_used_total": int(sum(client_train_rows_used.values())),
                "val_rows_used_total": int(sum(client_val_rows_used.values())),
                "test_rows": int(metrics_test["support_total"]),
                "global_test_rows_expected": global_test_rows_expected,
                "client_train_rows": client_train_rows,
                "client_val_rows": client_val_rows,
                "client_train_rows_used": client_train_rows_used,
                "client_val_rows_used": client_val_rows_used,
                "global_test_holdout": rel(self.scenario.global_test_npz, self.repo_root),
                "test_sent_to_clients": False,
            },
            "model": model_config,
            "training": {
                "strategy": "FedAvg",
                "framework": "Flower",
                "flower_version": fl.__version__,
                "runtime_mode": self.runtime_mode,
                "batch_size": int(self.training_cfg["batch_size"]),
                "local_epochs": int(self.training_cfg["local_epochs"]),
                "rounds_configured": rounds_configured,
                "rounds_completed": rounds_completed,
                "best_round": int(self.best_round),
                "selection_metric": "val_macro_f1",
                "selection_split": "server_validation",
            },
            "threshold": threshold_payload,
            "validation": {
                "metrics": metrics_val,
                "rows": int(len(val_result["y_true"])),
                "selection_split": "server_validation",
                "test_used_for_threshold": False,
                "scientific_significance": "low_for_smoke" if self.mode == "smoke" else "valid_for_run",
            },
            "test": {
                "metrics": metrics_test,
                "rows": int(metrics_test["support_total"]),
                "global_holdout": rel(self.scenario.global_test_npz, self.repo_root),
                "used_after_threshold_selection": True,
                "scientific_significance": "low_for_smoke" if self.mode == "smoke" else "valid_for_run",
            },
            "best_round": int(self.best_round),
            "best_val_macro_f1": float(self.best_macro_f1),
            "global_test_holdout": rel(self.scenario.global_test_npz, self.repo_root),
            "test_sent_to_clients": False,
            "max_samples_per_client": self.max_samples_per_client,
            "metrics_val": metrics_val,
            "metrics_test": metrics_test,
            "comparison_with_p4": comparison,
            "artifacts": artifacts,
            "figures": figures,
            "criteria": criteria,
            "warnings": warnings,
            "errors": errors,
            "scientific_significance": "low_for_smoke" if self.mode == "smoke" else "valid_for_run",
            "round_rows": self.round_rows,
            "run_console_log": rel(self.logs_dir / "run_console.log", self.repo_root),
        }
        manifest = {
            "run_id": self.run_id,
            "accepted": accepted,
            "flower_runtime": True,
            "run_summary": rel(self.artifacts_dir / "run_summary.json", self.repo_root),
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
        write_json(run_summary_path, summary)
        write_json(run_manifest_path, manifest)
        write_json(root_manifest_path, manifest)
        latest_summary_path = write_latest_run_summary(run_dir=self.run_dir, repo_root=self.repo_root, summary=summary)
        summary["latest_run_summary"] = rel(latest_summary_path, self.repo_root)
        latest_run_path = self.run_dir.parents[1] / "latest_run.json"
        for scenario_level_path in [latest_run_path, latest_summary_path]:
            if scenario_level_path.exists():
                scenario_level_rel = rel(scenario_level_path, self.repo_root)
                if scenario_level_rel not in summary["artifacts"]:
                    summary["artifacts"].append(scenario_level_rel)
        write_json(run_summary_path, summary)
        write_json(latest_summary_path, summary)
        self._server_log(
            f"Final evaluation on global test holdout | macro_f1={metrics_test['macro_f1']:.4f}"
        )
        return summary


def build_initial_parameters(config: dict[str, Any]) -> Parameters:
    """Build initial Flower parameters for tests and app setup."""

    return ndarrays_to_parameters(get_parameters(build_model(config["model"])))
