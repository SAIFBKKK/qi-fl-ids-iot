from __future__ import annotations

from typing import Any, Mapping

import mlflow

from src.tracking.run_naming import generate_experiment_display_name, generate_run_name


class MLflowRunLogger:
    """
    Thin MLflow wrapper used alongside BaselineArtifactTracker.

    Lifecycle:
        logger = MLflowRunLogger(experiment, config)
        logger.start()
        logger.log_round_metrics(server_round=1, fit_metrics={...}, eval_metrics={...})
        logger.finish(status="success", duration_sec=42.1)
    """

    def __init__(
        self,
        experiment: Mapping[str, Any],
        config: Mapping[str, Any],
        tracking_uri: str = "mlruns",
    ) -> None:
        self.experiment = dict(experiment)
        self.config = dict(config)
        self._run: mlflow.ActiveRun | None = None

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(generate_experiment_display_name(experiment))

    def start(self) -> None:
        run_name = generate_run_name(self.experiment)
        self._run = mlflow.start_run(run_name=run_name)

        mlflow.log_params({
            "experiment_name":    self.experiment.get("name"),
            "architecture":       self.experiment.get("architecture"),
            "fl_strategy":        self.experiment.get("fl_strategy"),
            "data_scenario":      self.experiment.get("data_scenario"),
            "imbalance_strategy": self.experiment.get("imbalance_strategy"),
            "num_rounds":         self.config.get("strategy", {}).get("num_rounds"),
            "num_clients":        self.config.get("scenario", {}).get("num_clients"),
            "local_epochs":       self.config.get("train", {}).get("local_epochs"),
            "batch_size":         self.config.get("train", {}).get("batch_size"),
            "learning_rate":      self.config.get("train", {}).get("learning_rate"),
            "proximal_mu":        self.config.get("train", {}).get("proximal_mu"),
            "feature_count":      self.config.get("dataset", {}).get("feature_count"),
            "num_classes":        self.config.get("dataset", {}).get("num_classes"),
            "seed":               self.config.get("project", {}).get("seed", 42),
        })

    def log_round_metrics(
        self,
        server_round: int,
        fit_metrics: Mapping[str, Any] | None = None,
        eval_metrics: Mapping[str, Any] | None = None,
        distributed_loss: float | None = None,
    ) -> None:
        if self._run is None:
            return

        step = int(server_round)
        scalar_fit = {
            k: float(v)
            for k, v in (fit_metrics or {}).items()
            if isinstance(v, (int, float))
        }
        scalar_eval = {
            k: float(v)
            for k, v in (eval_metrics or {}).items()
            if isinstance(v, (int, float))
        }

        if distributed_loss is not None:
            mlflow.log_metric("distributed_loss", float(distributed_loss), step=step)
        if scalar_fit:
            mlflow.log_metrics(scalar_fit, step=step)
        if scalar_eval:
            mlflow.log_metrics(scalar_eval, step=step)

    def finish(self, *, status: str, duration_sec: float) -> None:
        if self._run is None:
            return

        mlflow.log_metric("duration_sec", round(float(duration_sec), 2))
        mlflow.set_tag("status", status)
        mlflow.end_run(status="FINISHED" if status == "success" else "FAILED")
        self._run = None
