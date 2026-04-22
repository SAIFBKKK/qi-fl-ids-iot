from __future__ import annotations

import json
from datetime import datetime, timezone
from numbers import Number
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from flwr.server.history import History

from src.common.paths import OUTPUTS_DIR

FIT_METRIC_KEYS = (
    "train_loss_last",
    "train_time_sec",
    "update_size_bytes",
)

EVALUATE_METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "recall_macro",
    "benign_recall",
    "false_positive_rate",
    "rare_class_recall",
)

ROUND_REQUIRED_KEYS = (
    "distributed_loss",
    *FIT_METRIC_KEYS,
    *EVALUATE_METRIC_KEYS,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_numeric_metrics(
    metrics: Mapping[str, Any],
    allowed_keys: tuple[str, ...],
) -> dict[str, float]:
    coerced: dict[str, float] = {}
    for key in allowed_keys:
        value = metrics.get(key)
        if isinstance(value, Number):
            coerced[key] = float(value)
    return coerced


class BaselineArtifactTracker:
    def __init__(self, experiment: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        self.experiment = dict(experiment)
        self.config = dict(config)
        self._fit_history: dict[int, dict[str, float]] = {}
        self._evaluate_history: dict[int, dict[str, float]] = {}
        self._distributed_losses: dict[int, float] = {}
        self._lock = Lock()

    @property
    def report_dir(self) -> Path:
        return OUTPUTS_DIR / "reports" / "baselines" / str(self.experiment["name"])

    def record_fit_round(self, server_round: int, metrics: Mapping[str, Any]) -> None:
        with self._lock:
            self._fit_history[int(server_round)] = _coerce_numeric_metrics(
                metrics,
                FIT_METRIC_KEYS,
            )

    def record_evaluate_round(
        self,
        server_round: int,
        distributed_loss: float | None,
        metrics: Mapping[str, Any],
    ) -> None:
        round_idx = int(server_round)
        with self._lock:
            if distributed_loss is not None:
                self._distributed_losses[round_idx] = float(distributed_loss)
            self._evaluate_history[round_idx] = _coerce_numeric_metrics(
                metrics,
                EVALUATE_METRIC_KEYS,
            )

    def build_round_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            rounds = sorted(
                set(self._fit_history)
                | set(self._evaluate_history)
                | set(self._distributed_losses)
            )
            fit_history = {rnd: dict(values) for rnd, values in self._fit_history.items()}
            evaluate_history = {
                rnd: dict(values) for rnd, values in self._evaluate_history.items()
            }
            distributed_losses = dict(self._distributed_losses)

        rows: list[dict[str, Any]] = []
        for server_round in rounds:
            row: dict[str, Any] = {
                "round": int(server_round),
                "distributed_loss": distributed_losses.get(server_round),
            }
            fit_metrics = fit_history.get(server_round, {})
            evaluate_metrics = evaluate_history.get(server_round, {})

            for key in FIT_METRIC_KEYS:
                row[key] = fit_metrics.get(key)
            for key in EVALUATE_METRIC_KEYS:
                row[key] = evaluate_metrics.get(key)

            rows.append(row)

        return rows

    def to_history(self) -> History:
        history = History()

        with self._lock:
            distributed_losses = dict(self._distributed_losses)
            fit_history = {rnd: dict(values) for rnd, values in self._fit_history.items()}
            evaluate_history = {
                rnd: dict(values) for rnd, values in self._evaluate_history.items()
            }

        for server_round in sorted(distributed_losses):
            history.add_loss_distributed(server_round, distributed_losses[server_round])

        for server_round in sorted(fit_history):
            history.add_metrics_distributed_fit(server_round, fit_history[server_round])

        for server_round in sorted(evaluate_history):
            history.add_metrics_distributed(server_round, evaluate_history[server_round])

        return history

    def build_round_metrics_payload(self) -> dict[str, Any]:
        return {
            "generated_at": _utc_now_iso(),
            "experiment_name": self.experiment["name"],
            "rounds": self.build_round_rows(),
        }

    def _is_complete_round(self, row: Mapping[str, Any]) -> bool:
        return all(row.get(key) is not None for key in ROUND_REQUIRED_KEYS)

    def _completed_round_count(self, round_rows: list[dict[str, Any]]) -> int:
        return sum(1 for row in round_rows if self._is_complete_round(row))

    def _resolve_status(
        self,
        *,
        runtime_status: str,
        round_rows: list[dict[str, Any]],
    ) -> str:
        if runtime_status != "success":
            return runtime_status

        requested_rounds = int(self.config["strategy"]["num_rounds"])
        completed_rounds = self._completed_round_count(round_rows)
        if completed_rounds < requested_rounds:
            return "partial"

        return "success"

    def _find_last_complete_round(
        self,
        round_rows: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        for row in reversed(round_rows):
            if self._is_complete_round(row):
                return row
        return None

    def build_run_summary(
        self,
        *,
        status: str,
        duration_sec: float,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        round_rows = self.build_round_rows()
        completed_rounds = self._completed_round_count(round_rows)
        effective_status = self._resolve_status(
            runtime_status=status,
            round_rows=round_rows,
        )
        final_row = self._find_last_complete_round(round_rows) or {}

        summary: dict[str, Any] = {
            "generated_at": _utc_now_iso(),
            "experiment_name": self.experiment["name"],
            "status": effective_status,
            "duration_sec": round(float(duration_sec), 2),
            "rounds": int(len(round_rows)),
            "completed_rounds": int(completed_rounds),
            "requested_rounds": int(self.config["strategy"]["num_rounds"]),
            "architecture": self.experiment["architecture"],
            "fl_strategy": self.experiment["fl_strategy"],
            "data_scenario": self.experiment["data_scenario"],
            "imbalance_strategy": self.experiment["imbalance_strategy"],
            "num_clients": int(self.config["scenario"]["num_clients"]),
            "feature_count": int(self.config["dataset"]["feature_count"]),
            "num_classes": int(self.config["dataset"]["num_classes"]),
            "seed": int(self.config["project"].get("seed", 42)),
        }

        if error_message:
            summary["error_message"] = error_message

        if hasattr(self, "strategy") and hasattr(self.strategy, "best_round_info"):
            summary["best_round"] = self.strategy.best_round_info

        if final_row:
            for metric_name in ("distributed_loss",) + EVALUATE_METRIC_KEYS:
                value = final_row.get(metric_name)
                if value is not None:
                    summary[f"final_{metric_name}"] = value

        return summary

    def _build_observations(
        self,
        *,
        status: str,
        duration_sec: float,
        error_message: str | None = None,
    ) -> list[str]:
        round_rows = self.build_round_rows()
        observations: list[str] = []

        if status != "success":
            if error_message:
                observations.append(f"Le run a échoué avec l'erreur: {error_message}")
            return observations

        completed_rounds = self._completed_round_count(round_rows)
        requested_rounds = int(self.config["strategy"]["num_rounds"])
        effective_status = self._resolve_status(
            runtime_status=status,
            round_rows=round_rows,
        )

        observations.append(
            f"Le baseline a terminé {len(round_rows)} rounds distribués avec "
            f"{int(self.config['scenario']['num_clients'])} clients en "
            f"{round(float(duration_sec), 2)} secondes."
        )

        if effective_status == "partial":
            observations.append(
                f"Seulement {completed_rounds} rounds sur {requested_rounds} ont produit "
                "un jeu complet de métriques agrégées."
            )

        final_row = self._find_last_complete_round(round_rows)
        if final_row:
            macro_f1 = final_row.get("macro_f1")
            rare_recall = final_row.get("rare_class_recall")
            benign_recall = final_row.get("benign_recall")
            update_size_bytes = final_row.get("update_size_bytes")

            if macro_f1 is not None:
                observations.append(
                    f"Le macro-F1 distribué final agrégé est de {macro_f1:.4f}."
                )
            if rare_recall is not None:
                observations.append(
                    f"Le rappel final des classes rares agrégé est de {rare_recall:.4f}."
                )
            if benign_recall is not None:
                observations.append(
                    f"Le rappel final de la classe bénigne agrégé est de {benign_recall:.4f}."
                )
            if update_size_bytes is not None:
                observations.append(
                    f"La taille moyenne pondérée de mise à jour par client au dernier round "
                    f"est de {update_size_bytes:.2f} bytes."
                )

        observations.append(
            "Cette baseline sert de référence de comparaison pour FedProx, SCAFFOLD "
            "et les variantes avec client expert."
        )
        return observations

    def build_baseline_notes(
        self,
        *,
        status: str,
        duration_sec: float,
        error_message: str | None = None,
    ) -> str:
        summary = self.build_run_summary(
            status=status,
            duration_sec=duration_sec,
            error_message=error_message,
        )
        observations = self._build_observations(
            status=status,
            duration_sec=duration_sec,
            error_message=error_message,
        )

        notes_lines = [
            "# Baseline officielle",
            "",
            "## Résumé",
            f"- Experiment: {self.experiment['name']}",
            f"- Architecture: {self.experiment['architecture']}",
            f"- Strategy: {self.experiment['fl_strategy']}",
            f"- Scenario: {self.experiment['data_scenario']}",
            f"- Imbalance: {self.experiment['imbalance_strategy']}",
            f"- Status: {status}",
            f"- Effective status: {summary['status']}",
            f"- Duration sec: {summary['duration_sec']}",
            f"- Completed rounds: {summary['completed_rounds']} / {summary['requested_rounds']}",
            f"- Num clients: {summary['num_clients']}",
            f"- Feature count: {summary['feature_count']}",
            f"- Num classes: {summary['num_classes']}",
            f"- Seed: {summary['seed']}",
            "",
            "## Final metrics",
        ]

        for key in (
            "final_distributed_loss",
            "final_accuracy",
            "final_macro_f1",
            "final_recall_macro",
            "final_benign_recall",
            "final_false_positive_rate",
            "final_rare_class_recall",
        ):
            if key in summary:
                notes_lines.append(f"- {key}: {summary[key]}")

        notes_lines.extend(
            [
                "",
                "## Observations",
            ]
        )
        notes_lines.extend(f"- {observation}" for observation in observations)
        notes_lines.extend(
            [
                "",
                "## Configuration complète",
                "```json",
                json.dumps(
                    {
                        "experiment": self.experiment,
                        "config": self.config,
                    },
                    indent=2,
                ),
                "```",
            ]
        )
        return "\n".join(notes_lines) + "\n"

    def save_baseline_artifacts(
        self,
        *,
        status: str,
        duration_sec: float,
        error_message: str | None = None,
    ) -> None:
        report_dir = self.report_dir
        report_dir.mkdir(parents=True, exist_ok=True)

        run_summary = self.build_run_summary(
            status=status,
            duration_sec=duration_sec,
            error_message=error_message,
        )
        round_metrics = self.build_round_metrics_payload()
        baseline_notes = self.build_baseline_notes(
            status=status,
            duration_sec=duration_sec,
            error_message=error_message,
        )

        with (report_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(run_summary, handle, indent=2)

        with (report_dir / "round_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(round_metrics, handle, indent=2)

        with (report_dir / "baseline_notes.md").open("w", encoding="utf-8") as handle:
            handle.write(baseline_notes)
