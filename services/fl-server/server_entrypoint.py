"""Flower training dispatcher for the Compose training profile.

TRAINING_MODE=registry  FastAPI registry only — no Flower server.
TRAINING_MODE=mock      Registry + background mock Flower training thread.
TRAINING_MODE=real      Registry + background real scientific runner thread.

FastAPI on port 8080 is ALWAYS the main server regardless of TRAINING_MODE.
Mock/real modes add a daemon thread on top; they never replace the HTTP server.
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any

import flwr as fl
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from loguru import logger
from pydantic import BaseModel

from metrics import FLServerMetrics
from model_registry import ModelRegistry
from node_registry import NodeRegistry


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_NUM_ROUNDS = 10
DEFAULT_REAL_EXPERIMENT = "exp_v4_multitier_fedavg_normal_classweights"
DEFAULT_REAL_WORKDIR = "/app/experiments/fl-iot-ids-v3"
METRICS_PORT = 8000
TIERS = ["weak", "medium", "powerful"]

MODEL_FACTORY_PATH = Path(os.getenv("MODEL_FACTORY_PATH", "/artifacts/model_factory_30rounds"))

# ---------------------------------------------------------------------------
# Module-level singletons (initialised once, shared across all requests)
# ---------------------------------------------------------------------------

_fl_metrics = FLServerMetrics()
_node_registry = NodeRegistry()
_model_registry = ModelRegistry()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=os.getenv("LOG_LEVEL", "INFO"),
        serialize=os.getenv("LOG_FORMAT", "json").lower() == "json",
        backtrace=False,
        diagnose=False,
    )


configure_logging()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TierInfo(BaseModel):
    tier: str
    available: bool
    model_path: str
    scaler_path: str
    feature_names_path: str
    label_mapping_path: str
    config: dict
    size_bytes: int
    md5: str | None


class ModelsList(BaseModel):
    tiers: list[TierInfo]
    factory_path: str
    factory_available: bool


class TierMetadata(TierInfo):
    label_mapping_summary: dict


class NodeRegistrationRequest(BaseModel):
    node_id: str
    cpu_cores: int = 1
    ram_mb: int = 1024
    device_type: str = "docker_node"
    network_quality: str = "medium"
    battery_powered: bool = False
    tier_override: str | None = None


class NodeRegistrationResponse(BaseModel):
    node_id: str
    assigned_tier: str
    model_version: str
    model_source: str
    status: str


# ---------------------------------------------------------------------------
# Model Factory helpers
# ---------------------------------------------------------------------------

def _read_tier(tier_name: str) -> TierInfo | None:
    tier_dir = MODEL_FACTORY_PATH / tier_name
    config_path = tier_dir / "model_config.json"
    model_path = tier_dir / "global_model.pth"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    size = model_path.stat().st_size if model_path.exists() else 0
    md5 = hashlib.md5(model_path.read_bytes()).hexdigest()[:8] if model_path.exists() else None
    return TierInfo(
        tier=tier_name,
        available=True,
        model_path=str(model_path),
        scaler_path=str(tier_dir / "scaler.pkl"),
        feature_names_path=str(tier_dir / "feature_names.json"),
        label_mapping_path=str(tier_dir / "label_mapping.json"),
        config=config,
        size_bytes=size,
        md5=md5,
    )


def _list_factory_tiers() -> list[TierInfo]:
    if not MODEL_FACTORY_PATH.exists():
        return []
    result = []
    for name in TIERS:
        info = _read_tier(name)
        if info is not None:
            result.append(info)
    return result


# ---------------------------------------------------------------------------
# Prometheus metrics server (port 8000, thread daemon — R8: never merged into FastAPI)
# ---------------------------------------------------------------------------

def _update_registry_metrics() -> None:
    _fl_metrics.update_registered_nodes(
        total=_node_registry.total(),
        by_tier=_node_registry.counts_by_tier(),
    )


def _start_metrics_server() -> None:
    metrics_obj = _fl_metrics

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/metrics":
                body = metrics_obj.prometheus_text().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/health":
                body = json.dumps(
                    {"status": "ok", "service": "fl-server", **metrics_obj.snapshot()}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

    server = http.server.HTTPServer(("0.0.0.0", METRICS_PORT), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info("fl_metrics_server_started port={}", METRICS_PORT)


# ---------------------------------------------------------------------------
# FedAvg with Prometheus instrumentation (preserved exactly)
# ---------------------------------------------------------------------------

class FedAvgWithMetrics(FedAvg):
    def __init__(self, num_rounds: int = DEFAULT_NUM_ROUNDS, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._num_rounds = num_rounds
        self._round_start: float = time.time()

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Any],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        round_duration = time.time() - self._round_start

        accuracy = 0.0
        benign_recall = 0.0
        f1_macro = 0.0
        total_bandwidth = 0
        total_samples = 0

        for _client, fit_res in results:
            n = fit_res.num_examples
            total_samples += n
            m = fit_res.metrics or {}
            accuracy      += float(m.get("accuracy",      0.0)) * n
            benign_recall += float(m.get("benign_recall", 0.0)) * n
            f1_macro      += float(m.get("f1_macro",      0.0)) * n
            total_bandwidth += sum(
                len(t) for t in (fit_res.parameters.tensors if fit_res.parameters else [])
            )

        if total_samples > 0:
            accuracy      /= total_samples
            benign_recall /= total_samples
            f1_macro      /= total_samples

        aggregated = super().aggregate_fit(server_round, results, failures)

        _fl_metrics.update(
            round_num=server_round,
            accuracy=accuracy,
            benign_recall=benign_recall,
            f1_macro=f1_macro,
            duration=round_duration,
            active_clients=len(results),
            bandwidth_bytes=total_bandwidth,
            false_positive_rate=0.0,
        )

        self._round_start = time.time()
        return aggregated


# ---------------------------------------------------------------------------
# MLflow helpers (preserved exactly)
# ---------------------------------------------------------------------------

def read_int_env(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError:
        logger.critical("{} must be an integer, got {!r}", name, raw_value)
        sys.exit(1)
    if value < minimum:
        logger.critical("{} must be >= {}, got {}", name, minimum, value)
        sys.exit(1)
    return value


def start_mlflow_run(training_mode: str, num_rounds: int) -> Any:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.warning("MLFLOW_TRACKING_URI is not set; MLflow logging disabled")
        return nullcontext()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "p5_mock_fl_training"))
        run = mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME", "p5-mock-fl-server"))
        mlflow.log_param("training_mode", training_mode)
        mlflow.log_param("num_rounds", num_rounds)
        mlflow.log_param("strategy", "FedAvg")
        mlflow.log_metric("server_started", 1.0)
        return run
    except Exception as exc:  # pragma: no cover
        logger.warning("MLflow logging disabled after startup failure: {}", exc)
        return nullcontext()


def log_mlflow_metric(name: str, value: float) -> None:
    if mlflow.active_run() is None:
        return
    try:
        mlflow.log_metric(name, value)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not log {} to MLflow: {}", name, exc)


# ---------------------------------------------------------------------------
# Training runners (preserved exactly — run in daemon threads in mock/real modes)
# ---------------------------------------------------------------------------

def run_mock_training(training_mode: str) -> None:
    host = os.getenv("FL_SERVER_HOST", "0.0.0.0")
    port = read_int_env("FL_SERVER_PORT", 8080)
    num_rounds = read_int_env("FL_NUM_ROUNDS", DEFAULT_NUM_ROUNDS)

    server_address = f"{host}:{port}"
    logger.info(
        "Starting mock Flower training server on {} for {} rounds",
        server_address,
        num_rounds,
    )
    logger.info("P5 mock training profile validates orchestration, not scientific FL metrics")

    _fl_metrics.update(
        round_num=0,
        accuracy=0.0,
        benign_recall=0.0,
        f1_macro=0.0,
        duration=0.0,
        active_clients=0,
        bandwidth_bytes=0,
    )

    strategy = FedAvgWithMetrics(
        num_rounds=num_rounds,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )

    with start_mlflow_run(training_mode=training_mode, num_rounds=num_rounds):
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
        logger.info("Mock Flower training finished after {} rounds", num_rounds)
        log_mlflow_metric("training_finished", 1.0)


def run_real_training() -> None:
    experiment = os.getenv("REAL_FL_EXPERIMENT", DEFAULT_REAL_EXPERIMENT)
    rounds = read_int_env("REAL_FL_ROUNDS", 1)
    workdir = Path(os.getenv("REAL_FL_WORKDIR", DEFAULT_REAL_WORKDIR))
    tracking_uri = "http://mlflow:5000"
    artifact_root = "/app/experiments/fl-iot-ids-v3/outputs/mlruns"
    git_python_refresh = "quiet"

    if not workdir.exists():
        logger.critical(
            "Real FL workdir is missing: {}. Mount experiments/fl-iot-ids-v3 first.",
            workdir,
        )
        sys.exit(1)

    runner = workdir / "src" / "scripts" / "run_experiment.py"
    if not runner.exists():
        logger.critical("Real FL runner not found: {}", runner)
        sys.exit(1)

    command = [
        sys.executable,
        "/app/real_runner_wrapper.py",
        "--experiment", experiment,
        "--rounds", str(rounds),
    ]

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = tracking_uri
    env["MLFLOW_ARTIFACT_ROOT"] = artifact_root
    env["GIT_PYTHON_REFRESH"] = git_python_refresh
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(workdir) if not existing_pythonpath
        else f"{workdir}{os.pathsep}{existing_pythonpath}"
    )

    logger.info(
        "TRAINING_MODE=real: launching scientific runner experiment={} rounds={} workdir={}",
        experiment, rounds, workdir,
    )
    logger.info("MLFLOW_TRACKING_URI injected as {}", tracking_uri)
    logger.info("MLFLOW_ARTIFACT_ROOT injected as {}", artifact_root)

    result = subprocess.run(command, cwd=str(workdir), env=env, check=False)
    if result.returncode != 0:
        logger.critical("Scientific runner failed with exit code {}", result.returncode)
        sys.exit(result.returncode)

    logger.info("Scientific runner completed successfully")


# ---------------------------------------------------------------------------
# Thread launcher for mock/real modes
# ---------------------------------------------------------------------------

def _launch_training_thread(target_fn: Any, *args: Any) -> None:
    def _wrapper():
        try:
            _fl_metrics.set_training_thread_status(1)
            target_fn(*args)
        except Exception as exc:
            logger.error("Training thread crashed: {}", exc)
            _fl_metrics.set_training_thread_status(0)

    t = threading.Thread(target=_wrapper, daemon=True, name="fl-training")
    t.start()
    logger.info("fl_training_thread_launched target={}", target_fn.__name__)


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Prometheus metrics server must start first so Gauges exist before training thread writes them
    _start_metrics_server()

    # 2. Init registry gauge values
    _update_registry_metrics()

    # 3. KEEP_SERVER_ALIVE deprecation notice
    if os.getenv("KEEP_SERVER_ALIVE") is not None:
        logger.info("KEEP_SERVER_ALIVE is deprecated under FastAPI/uvicorn, ignored")

    # 4. Training mode dispatch
    training_mode = os.getenv("TRAINING_MODE", "registry").lower()
    if training_mode not in {"registry", "mock", "real"}:
        logger.warning(
            "Unknown TRAINING_MODE={!r}, falling back to registry", training_mode
        )
        training_mode = "registry"

    logger.info("fl_server_startup training_mode={}", training_mode)

    if training_mode == "mock":
        _launch_training_thread(run_mock_training, training_mode)
    elif training_mode == "real":
        _launch_training_thread(run_real_training)

    yield
    # Shutdown: daemon threads auto-terminated by Python runtime


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan, title="fl-server", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "fl-server",
        "version": "2.0",
        "training_mode": os.getenv("TRAINING_MODE", "registry"),
        "factory_available": MODEL_FACTORY_PATH.exists(),
        **_fl_metrics.snapshot(),
    }


@app.get("/nodes")
def list_nodes() -> dict:
    return {"nodes": _node_registry.list_nodes()}


@app.get("/models", response_model=ModelsList)
def list_models() -> ModelsList:
    tiers = _list_factory_tiers()
    return ModelsList(
        tiers=tiers,
        factory_path=str(MODEL_FACTORY_PATH),
        factory_available=MODEL_FACTORY_PATH.exists(),
    )


@app.get("/models/{tier}/metadata", response_model=TierMetadata)
def get_model_metadata(tier: str) -> TierMetadata:
    normalized = tier.strip().lower()
    if normalized not in TIERS:
        raise HTTPException(status_code=404, detail={"error": "unknown_tier", "tier": normalized})

    if not MODEL_FACTORY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail={"error": "factory_not_mounted", "path": str(MODEL_FACTORY_PATH)},
        )

    info = _read_tier(normalized)
    if info is None:
        raise HTTPException(status_code=404, detail={"error": "tier_bundle_missing", "tier": normalized})

    tier_dir = MODEL_FACTORY_PATH / normalized
    label_mapping_path = tier_dir / "label_mapping.json"
    label_mapping_summary: dict = {}
    if label_mapping_path.exists():
        mapping = json.loads(label_mapping_path.read_text(encoding="utf-8"))
        label_mapping_summary = {
            "num_classes": len(mapping),
            "classes": list(mapping.keys()) if isinstance(mapping, dict) else [],
        }

    return TierMetadata(**info.model_dump(), label_mapping_summary=label_mapping_summary)


@app.post("/nodes/register", response_model=NodeRegistrationResponse)
def register_node(body: NodeRegistrationRequest) -> NodeRegistrationResponse:
    payload = body.model_dump()
    try:
        node = _node_registry.register(payload)
        _update_registry_metrics()
        return NodeRegistrationResponse(
            node_id=node.node_id,
            assigned_tier=node.assigned_tier,
            model_version=node.model_version,
            model_source=node.model_source,
            status=node.status,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_entrypoint:app", host="0.0.0.0", port=8080, log_level="info")
