from __future__ import annotations

import pickle
import subprocess
import time
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import Parameters, Scalar, parameters_to_ndarrays
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.utils.mlflow_logger import MLflowRunLogger, normalize_tracking_uri
from src.common.config import load_yaml_config
from src.common.paths import ARTIFACTS_DIR
from src.common.paths import OUTPUTS_DIR

logger = logging.getLogger("run_server")

EXPERT_NODE_ID = "node3"

# Metric keys that cannot be cast to float (skip in weighted-average aggregation)
_NON_NUMERIC_KEYS = {"node_id", "scaffold_delta_c"}


def _resolve_tracking_uri(raw_uri: str) -> str:
    if raw_uri.startswith(("http://", "https://", "file:")):
        return raw_uri
    path = Path(raw_uri)
    if path.is_absolute():
        return normalize_tracking_uri(str(path))
    if raw_uri in {".", "./outputs/mlruns", "outputs/mlruns"}:
        return normalize_tracking_uri(str(OUTPUTS_DIR / "mlruns"))
    return normalize_tracking_uri(str((OUTPUTS_DIR / raw_uri).resolve()))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Metric aggregation helpers
# ---------------------------------------------------------------------------

def _params_nbytes(parameters: Parameters) -> int:
    return sum(p.nbytes for p in parameters_to_ndarrays(parameters))


def _aggregate_fit_metrics(
    metrics: List[Tuple[int, Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Aggregate per-client fit metrics.
      bytes_sent, bytes_received, fit_time_sec  → summed
      all other numeric metrics                 → weighted average by num_examples
      node_id, scaffold_delta_c                 → skipped (non-numeric)
    """
    if not metrics:
        return {}

    SUM_KEYS = {"bytes_sent", "bytes_received", "fit_time_sec"}
    total_examples = sum(n for n, _ in metrics)
    aggregated: Dict[str, float] = {}

    for num_examples, m in metrics:
        for key, value in m.items():
            if key in _NON_NUMERIC_KEYS:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if key in SUM_KEYS:
                aggregated[key] = aggregated.get(key, 0.0) + v
            else:
                w = num_examples / total_examples if total_examples > 0 else 0.0
                aggregated[key] = aggregated.get(key, 0.0) + v * w

    return aggregated


def _aggregate_eval_metrics(
    metrics: List[Tuple[int, Dict[str, Any]]],
) -> Dict[str, Any]:
    """Weighted average of evaluation metrics."""
    if not metrics:
        return {}
    total_examples = sum(n for n, _ in metrics)
    if total_examples == 0:
        return {}
    aggregated: Dict[str, float] = {}
    for num_examples, m in metrics:
        for key, value in m.items():
            aggregated[key] = aggregated.get(key, 0.0) + float(value) * num_examples
    return {k: v / total_examples for k, v in aggregated.items()}


# ---------------------------------------------------------------------------
# Base strategy: FedAvg + comm tracking + expert weighting
# ---------------------------------------------------------------------------

class CommTrackingFedAvg(FedAvg):
    """
    FedAvg with:
      - per-round communication cost tracking (bytes_up, bytes_down, latency)
      - optional expert-node weighting: node3's effective sample count is
        multiplied by expert_factor before aggregation
    """

    def __init__(
        self,
        *args: Any,
        mlflow_logger: MLflowRunLogger | None = None,
        num_clients: int = 3,
        expert_factor: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mlflow_logger = mlflow_logger
        self.num_clients = num_clients
        self.expert_factor = expert_factor
        self._round_start: Dict[int, float] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_start[server_round] = time.time()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        round_latency = time.time() - self._round_start.pop(server_round, time.time())

        original_results = results

        # Expert weighting — inflate node3's effective sample count before aggregation
        if self.expert_factor != 1.0:
            results = self._apply_expert_weights(server_round, results)

        aggregated_params, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        aggregated_metrics = _aggregate_fit_metrics(
            [(fit_res.num_examples, fit_res.metrics or {}) for _, fit_res in original_results]
        )

        if not results:
            return aggregated_params, aggregated_metrics

        bytes_up = sum(_params_nbytes(fit_res.parameters) for _, fit_res in results)
        bytes_down = (
            _params_nbytes(aggregated_params) * self.num_clients
            if aggregated_params is not None else 0
        )
        total_bytes = bytes_up + bytes_down

        logger.info(
            "Round %d | latency=%.2fs | bytes_up=%d | bytes_down=%d | total=%d",
            server_round, round_latency, bytes_up, bytes_down, total_bytes,
        )

        if self.mlflow_logger is not None:
            self.mlflow_logger.log_metrics(
                {
                    "bytes_up": float(bytes_up),
                    "bytes_down": float(bytes_down),
                    "total_bytes_round": float(total_bytes),
                    "round_latency_sec": round_latency,
                },
                step=server_round,
            )

        return aggregated_params, aggregated_metrics

    def _apply_expert_weights(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> List[Tuple[ClientProxy, FitRes]]:
        """Replace num_examples for node3 with num_examples * expert_factor."""
        adjusted = []
        for client_proxy, fit_res in results:
            node_id = (fit_res.metrics or {}).get("node_id", "")
            if node_id == EXPERT_NODE_ID:
                effective = int(fit_res.num_examples * self.expert_factor)
                fit_res = FitRes(
                    status=fit_res.status,
                    parameters=fit_res.parameters,
                    num_examples=effective,
                    metrics=fit_res.metrics,
                )
                logger.info(
                    "Round %d: %s expert weight ×%.1f → effective_samples=%d",
                    server_round, node_id, self.expert_factor, effective,
                )
            adjusted.append((client_proxy, fit_res))
        return adjusted


# ---------------------------------------------------------------------------
# SCAFFOLD strategy
# ---------------------------------------------------------------------------

class ScaffoldStrategy(CommTrackingFedAvg):
    """
    SCAFFOLD (Karimireddy et al., 2020) — corrects client drift via control variates.

    Server maintains a global control variate c_global saved to disk so that
    clients running on the same machine can read it without network serialisation.
    Clients compute delta_c each round and return it as bytes in fit metrics.

    Aggregation:
        c_global += (1 / N) * sum(delta_c_i)   N = number of clients
    """

    def __init__(self, *args: Any, artifacts_dir: Path = ARTIFACTS_DIR, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.artifacts_dir = artifacts_dir
        self.c_global: List[np.ndarray] | None = None

    def _c_global_path(self) -> Path:
        return self.artifacts_dir / "scaffold_c_global.pkl"

    def _save_c_global(self) -> None:
        if self.c_global is not None:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            with self._c_global_path().open("wb") as f:
                pickle.dump(self.c_global, f)

    def configure_fit(self, server_round, parameters, client_manager):
        """Initialise c_global to zeros on the first round, then save for clients."""
        if self.c_global is None and parameters is not None:
            ndarrays = parameters_to_ndarrays(parameters)
            self.c_global = [np.zeros_like(p, dtype=np.float32) for p in ndarrays]
            self._save_c_global()
            logger.info(
                "SCAFFOLD: initialised c_global to zeros (%d parameter tensors)", len(self.c_global)
            )
        elif self.c_global is not None:
            # Refresh on disk so clients always read the latest version
            self._save_c_global()

        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Collect delta_c from each client
        delta_c_list: List[List[np.ndarray]] = []
        for _, fit_res in results:
            metrics = fit_res.metrics or {}
            raw = metrics.get("scaffold_delta_c")
            if raw is not None:
                try:
                    delta_c = pickle.loads(raw)
                    delta_c_list.append(delta_c)
                except Exception as exc:
                    logger.warning("SCAFFOLD: failed to deserialise delta_c — %s", exc)

        # Update c_global: c_global += (1/N) * sum(delta_c_i)
        if delta_c_list and self.c_global is not None:
            n = len(delta_c_list)
            for i in range(len(self.c_global)):
                self.c_global[i] += (1.0 / n) * sum(dc[i] for dc in delta_c_list)
            self._save_c_global()
            delta_norms = [float(np.linalg.norm(dc[0])) for dc in delta_c_list]
            logger.info(
                "SCAFFOLD round %d: c_global updated from %d clients | delta_c[0] norms: %s",
                server_round, n, [f"{x:.5f}" for x in delta_norms],
            )

        # Normal FedAvg aggregation (comm tracking + expert weighting inherited)
        return super().aggregate_fit(server_round, results, failures)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Flower server for fl-iot-ids-v3")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--min-clients", type=int, default=None)
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["fedavg", "fedprox", "scaffold"],
        help="FL aggregation strategy (default: fedavg). "
             "fedprox uses standard FedAvg aggregation — proximal term is client-side only.",
    )
    parser.add_argument(
        "--expert-factor",
        type=float,
        default=None,
        help="Weight multiplier for node3 (expert) during aggregation (default: 1.0).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()

    cfg: Dict[str, Any] = {}
    if args.config is not None:
        cfg = load_yaml_config(args.config)

    server_cfg  = cfg.get("server", {})
    client_cfg  = cfg.get("client", {})
    mlflow_cfg  = cfg.get("mlflow", {})
    strategy_cfg = cfg.get("strategy_config", {})

    host        = args.host        or server_cfg.get("host",        "0.0.0.0")
    port        = args.port        or server_cfg.get("port",        8080)
    num_rounds  = args.num_rounds  or server_cfg.get("num_rounds",  10)
    min_clients = args.min_clients or server_cfg.get("min_clients", 3)
    strategy_name = (
        args.strategy if args.strategy is not None
        else server_cfg.get("strategy", "fedavg")
    )
    expert_factor = (
        args.expert_factor if args.expert_factor is not None
        else float(strategy_cfg.get("expert_factor", 1.0))
    )

    # MLflow setup
    mlflow_enabled = bool(mlflow_cfg.get("enabled", True))
    mlflow_logger: MLflowRunLogger | None = None

    if mlflow_enabled:
        run_name = mlflow_cfg.get("run_name", f"{strategy_name}-v3")
        mlflow_logger = MLflowRunLogger(
            tracking_uri=_resolve_tracking_uri(mlflow_cfg.get("tracking_uri", "./outputs/mlruns")),
            experiment_name=mlflow_cfg.get("experiment_name", "fl-iot-ids-v3"),
            run_name=run_name,
        )
        mlflow_logger.start()

    logger.info(
        "Starting Flower server | host=%s | port=%s | rounds=%d | clients=%d | "
        "strategy=%s | expert_factor=%.2f",
        host, port, num_rounds, min_clients, strategy_name, expert_factor,
    )

    if mlflow_logger is not None:
        mlflow_logger.log_params(
            {
                "model": "MLP",
                "architecture": "28->128->64->34",
                "strategy": strategy_name,
                "num_rounds": num_rounds,
                "num_clients": min_clients,
                "expert_factor": expert_factor,
                "local_epochs": client_cfg.get("local_epochs", 1),
                "batch_size": client_cfg.get("batch_size", 256),
                "learning_rate": client_cfg.get("learning_rate", 0.0005),
                "dataset": "CICIoT2023",
                "git_sha": _git_sha(),
                "class_weights_path": str(ARTIFACTS_DIR / f"class_weights_{server_cfg.get('scenario', 'normal_noniid')}.pkl"),
            }
        )

    # Build strategy
    shared_kwargs = dict(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        fit_metrics_aggregation_fn=_aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=_aggregate_eval_metrics,
        mlflow_logger=mlflow_logger,
        num_clients=min_clients,
        expert_factor=expert_factor,
    )

    if strategy_name == "scaffold":
        strategy = ScaffoldStrategy(**shared_kwargs, artifacts_dir=ARTIFACTS_DIR)
    else:
        # Both "fedavg" and "fedprox" use the same server-side aggregation.
        # FedProx proximal term is enforced on the client with --mu > 0.
        strategy = CommTrackingFedAvg(**shared_kwargs)

    start_time = time.time()

    try:
        history = fl.server.start_server(
            server_address=f"{host}:{port}",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )

        elapsed = time.time() - start_time
        logger.info("Training finished | total_time_sec=%.2f", elapsed)

        if mlflow_logger is not None:
            mlflow_logger.log_metrics({"total_time_sec": elapsed})

            for rnd, loss in getattr(history, "losses_distributed", []):
                mlflow_logger.log_metrics({"eval_loss": float(loss)}, step=rnd)

            eval_metric_map = {
                "accuracy":            "eval_accuracy",
                "f1_macro":            "eval_f1_macro",
                "recall_macro":        "eval_recall_macro",
                "precision_macro":     "eval_precision_macro",
                "benign_recall":       "eval_benign_recall",
                "false_positive_rate": "eval_false_positive_rate",
            }
            for metric_name, values in getattr(history, "metrics_distributed", {}).items():
                mlflow_key = eval_metric_map.get(metric_name, f"eval_{metric_name}")
                for rnd, val in values:
                    mlflow_logger.log_metrics({mlflow_key: float(val)}, step=rnd)

            for metric_name, values in getattr(history, "metrics_distributed_fit", {}).items():
                for rnd, val in values:
                    mlflow_logger.log_metrics({f"fit_{metric_name}": float(val)}, step=rnd)

            if args.config is not None:
                config_path = Path(__file__).resolve().parents[2] / args.config
                if config_path.exists():
                    mlflow_logger.log_artifact(config_path, artifact_path="configs")

    finally:
        if mlflow_logger is not None:
            mlflow_logger.end()


if __name__ == "__main__":
    main()
