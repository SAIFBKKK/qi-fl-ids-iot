from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


class MLflowRunLogger:
    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_name: str | None = None,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._active = False

    def start(self) -> None:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)
        self._active = True

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._active:
            return
        mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self._active:
            return
        cleaned: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                cleaned[key] = float(value)
            except (TypeError, ValueError):
                continue
        if cleaned:
            mlflow.log_metrics(cleaned, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if not self._active:
            return
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def end(self) -> None:
        if self._active:
            mlflow.end_run()
            self._active = False