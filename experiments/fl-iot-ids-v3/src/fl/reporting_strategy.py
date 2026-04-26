from __future__ import annotations

import json
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from flwr.common import EvaluateRes, FitIns, FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.common.logger import get_logger
from src.common.schemas import NodeProfile
from src.fl.node_profiler import NodeProfiler
from src.tracking.artifact_logger import BaselineArtifactTracker, build_mlflow_round_metrics


class ReportingFedAvg(FedAvg):
    def __init__(
        self,
        *,
        tracker: BaselineArtifactTracker | None = None,
        monitor_metric: str = "macro_f1",
        expert_node_id: str | None = None,
        expert_factor: float = 1.0,
        round_metric_logger: Callable[[int, dict[str, float]], None] | None = None,
        output_dir: Path | None = None,
        model_config: dict[str, Any] | None = None,
        node_profiler: NodeProfiler | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.logger = get_logger("reporting_strategy")
        self.tracker = tracker
        self.monitor_metric = monitor_metric
        self.expert_node_id = expert_node_id
        self.expert_factor = expert_factor
        self.round_metric_logger = round_metric_logger
        self._output_dir = output_dir
        self._model_config = model_config or {}
        self.node_profiler = node_profiler
        self._client_node_ids: dict[str, str] = {}
        self._best_metric: float = -math.inf
        self._best_round: int = 0
        self._best_params: Parameters | None = None
        self._latest_params: Parameters | None = None

    def _parameters_to_state_dict(self, parameters: Parameters) -> dict[str, torch.Tensor]:
        """Convertit flwr.Parameters → state_dict via une instance temporaire de MLPClassifier."""
        from src.model.network import MLPClassifier

        ndarrays = parameters_to_ndarrays(parameters)
        cfg = self._model_config
        tmp = MLPClassifier(
            input_dim=int(cfg.get("input_dim", 28)),
            num_classes=int(cfg.get("num_classes", 34)),
            hidden_dims=tuple(cfg.get("hidden_dims", [256, 128])),
            dropout=float(cfg.get("dropout", 0.2)),
        )
        keys = list(tmp.state_dict().keys())
        if len(keys) != len(ndarrays):
            raise ValueError(
                f"state_dict mismatch: {len(keys)} keys vs {len(ndarrays)} ndarrays. "
                "Vérifier que model_config correspond à l'architecture utilisée à l'entraînement."
            )
        return {k: torch.tensor(v) for k, v in zip(keys, ndarrays)}

    def _save_best_checkpoint(self, round_num: int, metrics: dict[str, Any]) -> None:
        """Sauvegarde le best checkpoint pour deployment."""
        if self._best_params is None or self._output_dir is None:
            return
        output_path = self._output_dir / "best_checkpoint.pth"
        try:
            state_dict = self._parameters_to_state_dict(self._best_params)
            checkpoint = {
                "state_dict": state_dict,
                "round": round_num,
                "macro_f1": metrics.get("macro_f1"),
                "benign_recall": metrics.get("benign_recall"),
                "false_positive_rate": metrics.get("false_positive_rate"),
                "saved_at": datetime.utcnow().isoformat(),
                "torch_version": torch.__version__,
            }
            self._output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, output_path)
            self.logger.info("Saved best checkpoint at round %d → %s", round_num, output_path)
        except Exception as exc:
            self.logger.error("Failed to save best checkpoint: %s", exc)

    @property
    def best_round_info(self) -> dict[str, object]:
        return {
            "best_round": self._best_round,
            "best_metric": self._best_metric,
            "monitor_metric": self.monitor_metric,
        }

    def _apply_expert_weighting(
        self,
        results: list[tuple[ClientProxy, FitRes]],
    ) -> list[tuple[ClientProxy, FitRes]]:
        if not self.expert_node_id or self.expert_factor == 1.0:
            return results

        adjusted: list[tuple[ClientProxy, FitRes]] = []
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics or {}
            node_id = metrics.get("node_id") or metrics.get("client_id")
            if node_id == self.expert_node_id:
                fit_res = FitRes(
                    status=fit_res.status,
                    parameters=fit_res.parameters,
                    num_examples=int(fit_res.num_examples * self.expert_factor),
                    metrics=fit_res.metrics,
                )
            adjusted.append((client_proxy, fit_res))
        return adjusted

    def _record_node_profiles(self, results: list[tuple[ClientProxy, FitRes]]) -> None:
        if self.node_profiler is None:
            return

        for client_proxy, fit_res in results:
            metrics = fit_res.metrics or {}
            node_id = metrics.get("node_id")
            if isinstance(node_id, str):
                self._client_node_ids[client_proxy.cid] = node_id

            raw_profile = metrics.get("node_profile_json")
            if raw_profile is None:
                continue
            if isinstance(raw_profile, bytes):
                raw_profile = raw_profile.decode("utf-8")
            if not isinstance(raw_profile, str):
                self.logger.warning(
                    "Ignoring non-string node_profile_json from client cid=%s",
                    client_proxy.cid,
                )
                continue

            try:
                profile = NodeProfile.from_dict(json.loads(raw_profile))
                assignment = self.node_profiler.assign_tier(profile)
                self._client_node_ids[client_proxy.cid] = profile.node_id
                self.logger.info(
                    "Node profile received | cid=%s | node_id=%s | assigned_tier=%s",
                    client_proxy.cid,
                    profile.node_id,
                    assignment.assigned_tier,
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self.logger.warning(
                    "Failed to parse node profile from client cid=%s: %s",
                    client_proxy.cid,
                    exc,
                )

    def _with_tier_config(
        self,
        configured: list[tuple[ClientProxy, FitIns]],
    ) -> list[tuple[ClientProxy, FitIns]]:
        if self.node_profiler is None:
            return configured

        updated: list[tuple[ClientProxy, FitIns]] = []
        for client_proxy, fit_ins in configured:
            config = dict(fit_ins.config)
            node_id = self._client_node_ids.get(client_proxy.cid)
            assignment = (
                self.node_profiler.get_assignment(node_id)
                if node_id is not None
                else None
            )
            if assignment is not None:
                config["assigned_tier"] = assignment.assigned_tier
                config["model_width"] = assignment.model_width
                config["tier_local_epochs"] = assignment.local_epochs
                config["tier_batch_size"] = assignment.batch_size
            updated.append((client_proxy, FitIns(parameters=fit_ins.parameters, config=config)))
        return updated

    def configure_fit(self, server_round, parameters, client_manager):
        configured = super().configure_fit(server_round, parameters, client_manager)
        return self._with_tier_config(configured)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        self._record_node_profiles(results)
        adjusted_results = self._apply_expert_weighting(results)
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round,
            results=adjusted_results,
            failures=failures,
        )
        # Report metrics with real sample counts, not expert-inflated weights.
        aggregation_fn = getattr(self, "fit_metrics_aggregation_fn", None)
        if aggregation_fn is not None:
            metrics_aggregated = aggregation_fn(
                [(fit_res.num_examples, fit_res.metrics or {}) for _, fit_res in results]
            )
        if self.tracker is not None:
            self.tracker.record_fit_round(server_round, metrics_aggregated)
        if self.round_metric_logger is not None:
            self.round_metric_logger(
                server_round,
                build_mlflow_round_metrics(metrics_aggregated),
            )
        self._latest_params = parameters_aggregated
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round=server_round,
            results=results,
            failures=failures,
        )
        if self.tracker is not None:
            self.tracker.record_evaluate_round(
                server_round=server_round,
                distributed_loss=loss_aggregated,
                metrics=metrics_aggregated,
            )
        if self.round_metric_logger is not None:
            self.round_metric_logger(
                server_round,
                build_mlflow_round_metrics(
                    metrics_aggregated,
                    distributed_loss=loss_aggregated,
                ),
            )

        metric_value = metrics_aggregated.get(self.monitor_metric) if metrics_aggregated else None
        if metric_value is not None and float(metric_value) > self._best_metric:
            self._best_metric = float(metric_value)
            self._best_round = server_round
            self._best_params = self._latest_params
            self._save_best_checkpoint(server_round, dict(metrics_aggregated or {}))

        return loss_aggregated, metrics_aggregated


class ReportingScaffold(ReportingFedAvg):
    """Reporting strategy with SCAFFOLD control-variate synchronization."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.c_global: list[np.ndarray] | None = None

    @staticmethod
    def _mean_norm(arrays: list[np.ndarray]) -> float:
        if not arrays:
            return 0.0
        return float(np.mean([np.linalg.norm(np.asarray(array)) for array in arrays]))

    def configure_fit(self, server_round, parameters, client_manager):
        if self.c_global is None:
            ndarrays = parameters_to_ndarrays(parameters)
            self.c_global = [np.zeros_like(param, dtype=np.float32) for param in ndarrays]

        configured = super().configure_fit(server_round, parameters, client_manager)
        payload = pickle.dumps(self.c_global)
        updated = []
        for client_proxy, fit_ins in configured:
            config = dict(fit_ins.config)
            config["scaffold_c_global"] = payload
            updated.append(
                (
                    client_proxy,
                    FitIns(parameters=fit_ins.parameters, config=config),
                )
            )
        return updated

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        delta_c_list: list[list[np.ndarray]] = []
        for _, fit_res in results:
            raw_delta = (fit_res.metrics or {}).get("scaffold_delta_c")
            if raw_delta is None:
                raise ValueError(
                    "SCAFFOLD client result missing scaffold_delta_c. "
                    "All scaffold clients must return control-variate deltas."
                )
            delta_c_list.append(pickle.loads(raw_delta))

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
        )

        if self.c_global is not None and delta_c_list:
            for idx in range(len(self.c_global)):
                self.c_global[idx] = self.c_global[idx] + (
                    sum(delta[idx] for delta in delta_c_list) / len(delta_c_list)
                )
            if self.round_metric_logger is not None:
                self.round_metric_logger(
                    server_round,
                    {"scaffold/c_global_norm": self._mean_norm(self.c_global)},
                )

        return parameters_aggregated, metrics_aggregated
