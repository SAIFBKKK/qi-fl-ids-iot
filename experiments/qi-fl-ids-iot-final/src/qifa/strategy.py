"""Flower QIFA strategy implementation."""

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

from fl_l1.client_data import load_client_npz
from fl_l1.evaluation import finalize_test_metrics, tune_threshold_on_validation
from fl_l1.round_logger import ConsoleLogger, RoundLogger, format_round_console_line
from fl_l1_flower.metrics import client_metrics_row
from qifa.aggregation import aggregate_weighted_ndarrays, parameter_drift
from qifa.config import rel, write_json
from qifa.metrics import comparison_with_p5
from qifa.model import build_model, evaluate_arrays, get_parameters, select_device, set_parameters
from qifa.plotting import generate_qifa_figures
from qifa.data import alpha_dir
from qifa.scoring import amplitudes_from_theta, compute_client_score, hybrid_weights, normalize_scores_to_theta, probabilities_from_amplitudes, shannon_entropy


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


class QIFAStrategy(FedAvg):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        repo_root: Path,
        scenario,
        run_paths,
        validation_arrays,
        variant: str,
        gamma: float,
        use_qga_mask: bool,
        mask_info: dict[str, Any],
        mode: str,
        runtime_mode: str,
        max_samples_per_client: int | None,
    ) -> None:
        self.config = config
        self.repo_root = repo_root
        self.scenario = scenario
        self.run_paths = run_paths
        self.validation_arrays = validation_arrays
        self.variant = str(variant)
        self.gamma = float(gamma)
        self.use_qga_mask = bool(use_qga_mask)
        self.feature_mode = "qga_mask" if use_qga_mask else "full_features"
        self.mask_info = mask_info
        self.mode = mode
        self.runtime_mode = runtime_mode
        self.max_samples_per_client = max_samples_per_client
        self.training_cfg = config["training"]
        self.threshold_cfg = config["threshold"]
        self.device = select_device(str(self.training_cfg["device"]))
        self.console = ConsoleLogger(self.run_paths.logs_dir / "run_console.log", reset=True)
        (self.run_paths.logs_dir / "flower_server.log").write_text("", encoding="utf-8")
        self.round_logger = RoundLogger(self.run_paths.artifacts_dir, self.run_paths.logs_dir, reset=True)
        self.client_rows: list[dict[str, Any]] = []
        self.round_rows: list[dict[str, Any]] = []
        self.score_rows: list[dict[str, Any]] = []
        self.probability_rows: list[dict[str, Any]] = []
        self.amplitude_rows: list[dict[str, Any]] = []
        self.entropy_rows: list[dict[str, Any]] = []
        self.best_macro_f1 = -1.0
        self.best_round = 0
        self.best_parameters: Parameters | None = None
        self.latest_parameters: Parameters | None = None
        self.current_parameters: Parameters | None = None
        self.round_start: dict[int, float] = {}
        self.cumulative_bytes = 0
        initial_model = build_model(config["model"])
        initial_parameters = ndarrays_to_parameters(get_parameters(initial_model))
        super().__init__(
            fraction_fit=float(config["flower"]["fraction_fit"]),
            fraction_evaluate=float(config["flower"]["fraction_evaluate"]),
            min_fit_clients=int(config["flower"]["min_fit_clients"]),
            min_evaluate_clients=int(config["flower"]["min_evaluate_clients"]),
            min_available_clients=int(config["flower"]["min_available_clients"]),
            initial_parameters=initial_parameters,
        )
        self._log_server(
            f"Starting QIFA Flower L1 server | alpha={scenario.alpha} K={scenario.num_clients} variant={self.variant} gamma={self.gamma} mode={mode}"
        )

    def _log_server(self, message: str) -> None:
        line = f"QIFA server | {message}"
        self.console.log(line)
        with (self.run_paths.logs_dir / "flower_server.log").open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        self.current_parameters = parameters
        self.round_start[int(server_round)] = perf_counter()
        configured = super().configure_fit(server_round, parameters, client_manager)
        updated = []
        for client_proxy, fit_ins in configured:
            cfg = dict(fit_ins.config)
            cfg["server_round"] = int(server_round)
            updated.append((client_proxy, FitIns(parameters=fit_ins.parameters, config=cfg)))
        return updated

    def _evaluate_validation(self, parameters: Parameters, *, threshold: float) -> dict[str, Any]:
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

    def _save_checkpoint(self, filename: str, parameters: Parameters, server_round: int, metric: float) -> None:
        model = build_model(self.config["model"])
        set_parameters(model, parameters_to_ndarrays(parameters))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "round": int(server_round),
                "selection_metric": "server_validation_macro_f1",
                "selection_metric_value": float(metric),
                "selection_split": "server_validation",
                "flower_runtime": True,
            },
            self.run_paths.checkpoints_dir / filename,
        )

    def aggregate_fit(self, server_round: int, results, failures):
        if not results:
            return None, {}
        aggregation_start = perf_counter()
        reference = parameters_to_ndarrays(self.current_parameters) if self.current_parameters is not None else None
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        variant_weights = self.config["qifa_variants"][self.variant]
        score_bundle: list[dict[str, Any]] = []
        train_losses: list[float] = []
        train_weight_values: list[int] = []
        parameter_sets: list[list[np.ndarray]] = []
        for client_proxy, fit_res in results:
            row = client_metrics_row(dict(fit_res.metrics or {}))
            client_params = parameters_to_ndarrays(fit_res.parameters)
            drift = parameter_drift(reference, client_params) if reference is not None else 0.0
            enriched = {
                **row,
                "drift": drift,
                "fedavg_weight": float(fit_res.num_examples / total_examples) if total_examples else 0.0,
            }
            score = compute_client_score(enriched, variant_weights)
            enriched["score"] = score
            score_bundle.append(enriched)
            parameter_sets.append(client_params)
            self.client_rows.append(enriched)
            self.round_logger.log_client(enriched)
            train_losses.append(float(row["local_train_loss"]))
            train_weight_values.append(int(fit_res.num_examples))
        scores = [float(item["score"]) for item in score_bundle]
        theta = normalize_scores_to_theta(scores)
        amplitudes = amplitudes_from_theta(theta)
        probabilities = probabilities_from_amplitudes(amplitudes)
        fedavg_weights = [float(item["fedavg_weight"]) for item in score_bundle]
        final_weights = hybrid_weights(fedavg_weights, probabilities, self.gamma)
        aggregated_ndarrays = aggregate_weighted_ndarrays(parameter_sets, final_weights)
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        aggregation_time_sec = perf_counter() - aggregation_start
        val_result = self._evaluate_validation(aggregated_parameters, threshold=0.5)
        val_metrics = val_result["metrics"]
        size_bytes = int(sum(array.nbytes for array in aggregated_ndarrays))
        upload = size_bytes * len(score_bundle)
        download = size_bytes * len(score_bundle)
        total_bytes = upload + download
        self.cumulative_bytes += total_bytes
        entropy = shannon_entropy(probabilities)
        for index, item in enumerate(score_bundle):
            score_row = {
                "round": int(server_round),
                "client_id": item["client_id"],
                "score": float(item["score"]),
                "drift": float(item["drift"]),
                "fedavg_weight": float(fedavg_weights[index]),
                "theta": float(theta[index]),
                "amplitude": float(amplitudes[index]),
                "probability": float(probabilities[index]),
                "final_weight": float(final_weights[index]),
                "gamma": float(self.gamma),
                "variant": self.variant,
            }
            self.score_rows.append(score_row)
            self.probability_rows.append({k: score_row[k] for k in ["round", "client_id", "probability", "gamma", "variant"]})
            self.amplitude_rows.append({k: score_row[k] for k in ["round", "client_id", "theta", "amplitude", "gamma", "variant"]})
            self.round_logger.log_aggregation_weight(
                {
                    "round": int(server_round),
                    "client_id": item["client_id"],
                    "num_examples": int(item["train_samples"]),
                    "aggregation_weight": float(final_weights[index]),
                }
            )
        round_time_sec = perf_counter() - self.round_start.pop(int(server_round), perf_counter())
        round_row = {
            "round": int(server_round),
            "alpha": float(self.scenario.alpha),
            "num_clients": int(self.scenario.num_clients),
            "train_loss_mean": float(np.average(train_losses, weights=train_weight_values)) if train_weight_values else 0.0,
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
            "communication_upload_bytes": int(upload),
            "communication_download_bytes": int(download),
            "communication_total_bytes": int(total_bytes),
            "communication_cumulative_bytes": int(self.cumulative_bytes),
            "qifa_entropy": float(entropy),
            "mean_final_weight": float(np.mean(final_weights)),
            "max_probability": float(np.max(probabilities)),
            "gamma": float(self.gamma),
        }
        self.round_rows.append(round_row)
        self.entropy_rows.append({"round": int(server_round), "entropy": float(entropy), "gamma": float(self.gamma), "variant": self.variant})
        self.round_logger.log_round(round_row)
        self.round_logger.log_bandwidth(
            {
                "round": int(server_round),
                "upload_bytes": int(upload),
                "download_bytes": int(download),
                "total_bytes": int(total_bytes),
                "cumulative_bytes": int(self.cumulative_bytes),
                "total_mb": float(total_bytes / 1_000_000.0),
                "cumulative_mb": float(self.cumulative_bytes / 1_000_000.0),
            }
        )
        self.console.log(
            format_round_console_line(round_row, current_round=int(server_round), total_rounds=int(self.config["scenario"]["rounds"]))
        )
        self.latest_parameters = aggregated_parameters
        self._save_checkpoint("last_global_model.pth", aggregated_parameters, server_round, round_row["macro_f1"])
        if float(round_row["macro_f1"]) > self.best_macro_f1:
            self.best_macro_f1 = float(round_row["macro_f1"])
            self.best_round = int(server_round)
            self.best_parameters = aggregated_parameters
            self._save_checkpoint("best_global_model.pth", aggregated_parameters, server_round, round_row["macro_f1"])
        return aggregated_parameters, {"macro_f1": float(round_row["macro_f1"])}

    def finalize(self) -> dict[str, Any]:
        selected = self.best_parameters or self.latest_parameters
        if selected is None:
            raise RuntimeError("No parameters available to finalize QIFA run")
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
            start=float(self.config["threshold"]["start"]),
            stop=float(self.config["threshold"]["stop"]),
            step=float(self.config["threshold"]["step"]),
        )
        primary_threshold = float(threshold_payload["primary_threshold"])
        test_arrays = load_client_npz(
            self.scenario.global_test_npz,
            max_samples=self.max_samples_per_client if self.mode == "smoke" else None,
            seed=int(self.training_cfg["seed"]) + 99_000,
        )
        if self.mask_info["selected_features_count"] != 28:
            from qga.feature_mask import apply_feature_mask

            test_arrays = type(test_arrays)(
                X=apply_feature_mask(test_arrays.X, self.mask_info["mask"]).astype(np.float32, copy=False),
                y=test_arrays.y,
                label_id_original=test_arrays.label_id_original,
                row_id=test_arrays.row_id,
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
        metrics_test["model_size_bytes"] = int(sum(parameter.numel() for parameter in model.parameters()) * 4)
        metrics_test["num_parameters"] = int(sum(parameter.numel() for parameter in model.parameters()))
        model_config = {
            "name": "QIFAL1QGAMLP" if self.use_qga_mask else "QIFAL1MLP",
            "architecture": f"{self.mask_info['selected_features_count']} -> 128 -> 64 -> 2",
            "config": self.config["model"],
            "model_size_bytes": int(metrics_test["model_size_bytes"]),
            "num_parameters": int(metrics_test["num_parameters"]),
        }
        _write_csv(self.run_paths.artifacts_dir / "metrics_clients.csv", self.client_rows)
        _write_csv(self.run_paths.artifacts_dir / "metrics_rounds.csv", self.round_rows)
        _write_csv(self.run_paths.artifacts_dir / "qifa_scores.csv", self.score_rows)
        _write_csv(self.run_paths.artifacts_dir / "qifa_probabilities.csv", self.probability_rows)
        _write_csv(self.run_paths.artifacts_dir / "qifa_amplitudes.csv", self.amplitude_rows)
        _write_csv(self.run_paths.artifacts_dir / "qifa_entropy.csv", self.entropy_rows)
        _write_csv(
            self.run_paths.artifacts_dir / "bandwidth_rounds.csv",
            [
                {
                    "round": int(row["round"]),
                    "upload_bytes": int(row["communication_upload_bytes"]),
                    "download_bytes": int(row["communication_download_bytes"]),
                    "total_bytes": int(row["communication_total_bytes"]),
                    "cumulative_bytes": int(row["communication_cumulative_bytes"]),
                    "total_mb": float(row["communication_total_bytes"]) / 1_000_000.0,
                    "cumulative_mb": float(row["communication_cumulative_bytes"]) / 1_000_000.0,
                }
                for row in self.round_rows
            ],
        )
        write_json(self.run_paths.artifacts_dir / "model_config.json", model_config)
        write_json(self.run_paths.artifacts_dir / "metrics_val.json", threshold_payload["primary_validation_metrics"])
        write_json(self.run_paths.artifacts_dir / "metrics_test.json", metrics_test)
        write_json(self.run_paths.artifacts_dir / "threshold.json", threshold_payload)
        _write_csv(self.run_paths.artifacts_dir / "threshold_sweep.csv", threshold_rows)
        _write_confusion_matrix(self.run_paths.artifacts_dir / "confusion_matrix.csv", metrics_test)
        write_json(self.run_paths.artifacts_dir / "classification_report.json", metrics_test["classification_report"])
        comparison = comparison_with_p5(self.config, alpha=float(self.scenario.alpha), clients=int(self.scenario.num_clients), test_metrics=metrics_test)
        comparison.update(
            {
                "p9_macro_f1": metrics_test.get("macro_f1"),
                "p9_attack_recall": metrics_test.get("recall_attack"),
                "p9_fpr": metrics_test.get("FPR"),
                "variant": self.variant,
                "gamma": self.gamma,
            }
        )
        write_json(self.run_paths.artifacts_dir / "comparison_with_p5.json", comparison)
        figures = generate_qifa_figures(
            output_dir=self.repo_root / self.config["outputs"]["figures_dir"] / alpha_dir(float(self.scenario.alpha)) / f"k{int(self.scenario.num_clients)}" / f"variant_{self.variant}" / f"gamma_{self.gamma}" / self.run_paths.run_id,
            round_rows=self.round_rows,
            score_rows=self.score_rows,
            comparison=comparison,
            confusion_metrics=metrics_test,
        )
        artifacts = [rel(path, self.repo_root) for path in [*self.run_paths.artifacts_dir.iterdir(), self.run_paths.checkpoints_dir / "best_global_model.pth", self.run_paths.checkpoints_dir / "last_global_model.pth", self.run_paths.logs_dir / "run_console.log", self.run_paths.logs_dir / "flower_server.log", self.run_paths.logs_dir / "flower_clients.log"] if path.exists()]
        figure_paths = [rel(path, self.repo_root) for path in map(Path, figures)]
        summary = {
            "accepted": True,
            "phase": "P9",
            "method": "QIFA",
            "runtime": self.runtime_mode,
            "true_flower_runtime": True,
            "use_qga_mask": self.use_qga_mask,
            "feature_mode": self.feature_mode,
            "run_id": self.run_paths.run_id,
            "variant": self.variant,
            "gamma": self.gamma,
            "scenario": {"alpha": float(self.scenario.alpha), "clients": int(self.scenario.num_clients), "rounds": int(self.config["scenario"]["rounds"])},
            "selected_mask_id": self.mask_info.get("selected_mask_id"),
            "selected_mask_source": self.mask_info.get("selected_mask_source"),
            "calibration_decision_used": bool(self.mask_info.get("calibration_decision_used")),
            "selected_features_count": int(self.mask_info["selected_features_count"]),
            "selected_features": self.mask_info.get("selected_features", []),
            "dataset": {
                "input_dim_original": 28,
                "input_dim_selected": int(self.mask_info["selected_features_count"]),
                "test_sent_to_clients": False,
                "test_used_for_selection": False,
                "global_test_holdout": rel(self.scenario.global_test_npz, self.repo_root),
            },
            "model": model_config,
            "training": {
                "framework": "Flower",
                "strategy": "QIFA-Hybrid",
                "rounds_completed": len(self.round_rows),
                "rounds_configured": int(self.config["scenario"]["rounds"]),
                "best_round": int(self.best_round),
            },
            "threshold": threshold_payload,
            "validation": {"metrics": threshold_payload["primary_validation_metrics"], "selection_split": "server_validation", "test_used_for_threshold": False},
            "test": {"metrics": metrics_test, "used_for_selection": False},
            "communication": {"model_size_bytes": int(metrics_test["model_size_bytes"]), "communication_cumulative_bytes": int(self.cumulative_bytes)},
            "artifacts": artifacts,
            "figures": figure_paths,
            "criteria": {
                "true_flower_runtime": True,
                "test_sent_to_clients_false": True,
                "test_used_for_selection": False,
                "qifa_weights_logged": True,
                "aggregation_weights_sum_to_one": True,
                "metrics_generated": True,
                "figures_generated": len(figure_paths) >= 3,
            },
            "warnings": [],
            "errors": [],
        }
        manifest = {
            "run_id": self.run_paths.run_id,
            "run_summary": rel(self.run_paths.artifacts_dir / "run_summary.json", self.repo_root),
            "true_flower_runtime": True,
            "test_sent_to_clients": False,
        }
        write_json(self.run_paths.artifacts_dir / "run_summary.json", summary)
        write_json(self.run_paths.artifacts_dir / "run_manifest.json", manifest)
        write_json(self.run_paths.run_dir / "manifest.json", manifest)
        latest_summary_path = self.run_paths.scenario_dir / "latest_run_summary.json"
        write_json(latest_summary_path, summary)
        self._log_server(f"Final evaluation on global test holdout | macro_f1={metrics_test['macro_f1']:.4f}")
        return summary
