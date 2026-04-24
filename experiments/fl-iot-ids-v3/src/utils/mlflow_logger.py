from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import quote, urlparse

import mlflow


_SUPPORTED_URI_SCHEMES = {
    "",
    "file",
    "http",
    "https",
    "sqlite",
    "postgresql",
    "mysql",
    "mssql",
    "databricks",
    "databricks-uc",
    "uc",
}
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def normalize_tracking_uri(tracking_uri: str) -> str:
    """
    Normalize local MLflow paths into URIs that work on Windows and POSIX.

    MLflow interprets a raw Windows path such as ``C:\\runs\\mlruns`` as the
    unsupported URI scheme ``c``. Converting it to ``file:///C:/runs/mlruns``
    keeps local-file tracking portable across platforms.
    """
    parsed = urlparse(tracking_uri)
    if parsed.scheme.lower() in _SUPPORTED_URI_SCHEMES and parsed.scheme != "":
        return tracking_uri

    if _WINDOWS_DRIVE_RE.match(tracking_uri):
        windows_path = PureWindowsPath(tracking_uri).as_posix()
        return f"file:///{quote(windows_path, safe=':/')}"

    path = Path(tracking_uri)
    if path.is_absolute():
        return path.resolve().as_uri()

    return tracking_uri


class MLflowRunLogger:
    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_name: str | None = None,
    ) -> None:
        self.tracking_uri = normalize_tracking_uri(tracking_uri)
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
