"""CSV/JSONL round logging for P5 FedAvg L1."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRICS_ROUNDS_COLUMNS = [
    "round",
    "alpha",
    "num_clients",
    "train_loss_mean",
    "val_loss_mean",
    "accuracy",
    "precision",
    "recall",
    "macro_f1",
    "weighted_f1",
    "attack_precision",
    "attack_recall",
    "attack_f1",
    "FPR",
    "FNR",
    "TP",
    "TN",
    "FP",
    "FN",
    "round_time_sec",
    "aggregation_time_sec",
    "model_size_bytes",
    "communication_upload_bytes",
    "communication_download_bytes",
    "communication_total_bytes",
    "communication_cumulative_bytes",
]

METRICS_CLIENTS_COLUMNS = [
    "round",
    "client_id",
    "train_samples",
    "val_samples",
    "normal_count",
    "attack_count",
    "local_train_loss",
    "local_val_loss",
    "local_accuracy",
    "local_macro_f1",
    "local_attack_recall",
    "local_fpr",
    "fit_time_sec",
    "eval_time_sec",
    "upload_bytes",
    "download_bytes",
]

BANDWIDTH_COLUMNS = [
    "round",
    "upload_bytes",
    "download_bytes",
    "total_bytes",
    "cumulative_bytes",
    "total_mb",
    "cumulative_mb",
]

AGGREGATION_COLUMNS = ["round", "client_id", "num_examples", "aggregation_weight"]


def _float(value: Any, *, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def format_round_console_line(row: dict[str, Any], *, current_round: int, total_rounds: int) -> str:
    """Format the compact per-round console line requested for P5.1."""

    return (
        f"[Round {current_round:02d}/{total_rounds:02d}] "
        f"alpha={_float(row['alpha'], digits=1)} "
        f"K={int(row['num_clients'])} "
        f"loss={_float(row['train_loss_mean'])} "
        f"val_loss={_float(row['val_loss_mean'])} "
        f"macro_f1={_float(row['macro_f1'])} "
        f"attack_recall={_float(row['attack_recall'])} "
        f"FPR={_float(row['FPR'])} "
        f"TP={int(row['TP'])} TN={int(row['TN'])} FP={int(row['FP'])} FN={int(row['FN'])} "
        f"time={_float(row['round_time_sec'], digits=2)}s "
        f"model_bytes={int(row['model_size_bytes'])} "
        f"upload={int(row['communication_upload_bytes'])} "
        f"download={int(row['communication_download_bytes'])} "
        f"bytes={int(row['communication_total_bytes'])} "
        f"cum={int(row['communication_cumulative_bytes'])}"
    )


def format_client_console_line(row: dict[str, Any]) -> str:
    """Format an optional verbose client line."""

    return (
        f"[Client {row['client_id']}] "
        f"train={int(row['train_samples'])} "
        f"val={int(row['val_samples'])} "
        f"loss={_float(row['local_train_loss'])} "
        f"val_loss={_float(row['local_val_loss'])} "
        f"macro_f1={_float(row['local_macro_f1'])} "
        f"attack_recall={_float(row['local_attack_recall'])} "
        f"FPR={_float(row['local_fpr'])} "
        f"fit={_float(row['fit_time_sec'], digits=2)}s "
        f"upload={int(row['upload_bytes'])} "
        f"download={int(row['download_bytes'])}"
    )


def _append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class RoundLogger:
    """Append-only logger for FL rounds."""

    def __init__(self, artifacts_dir: Path, logs_dir: Path, *, reset: bool = False) -> None:
        self.artifacts_dir = artifacts_dir
        self.logs_dir = logs_dir
        self.metrics_rounds_path = artifacts_dir / "metrics_rounds.csv"
        self.metrics_clients_path = artifacts_dir / "metrics_clients.csv"
        self.bandwidth_rounds_path = artifacts_dir / "bandwidth_rounds.csv"
        self.aggregation_weights_path = artifacts_dir / "aggregation_weights.csv"
        self.events_jsonl_path = logs_dir / "events.jsonl"
        if reset:
            for path in [
                self.metrics_rounds_path,
                self.metrics_clients_path,
                self.bandwidth_rounds_path,
                self.aggregation_weights_path,
                self.events_jsonl_path,
            ]:
                if path.exists():
                    path.unlink()

    def log_round(self, row: dict[str, Any]) -> None:
        _append_csv(self.metrics_rounds_path, row, METRICS_ROUNDS_COLUMNS)

    def log_client(self, row: dict[str, Any]) -> None:
        _append_csv(self.metrics_clients_path, row, METRICS_CLIENTS_COLUMNS)

    def log_bandwidth(self, row: dict[str, Any]) -> None:
        _append_csv(self.bandwidth_rounds_path, row, BANDWIDTH_COLUMNS)

    def log_aggregation_weight(self, row: dict[str, Any]) -> None:
        _append_csv(self.aggregation_weights_path, row, AGGREGATION_COLUMNS)

    def log_event(self, payload: dict[str, Any]) -> None:
        self.events_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


class ConsoleLogger:
    """Mirror concise runtime logs to terminal and run_console.log."""

    def __init__(self, path: Path, *, echo: bool = True, reset: bool = True) -> None:
        self.path = path
        self.echo = bool(echo)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"{timestamp} | {message}"
        if self.echo:
            print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
