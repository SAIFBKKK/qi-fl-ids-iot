from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, Response
from loguru import logger
from pydantic import BaseModel, Field

from metrics import (
    QGA_LAST_SCORE,
    QGA_OPTIMIZATION_LATENCY_SECONDS,
    QGA_REQUESTS_TOTAL,
    QGA_SERVICE_STATUS,
    prometheus_text,
)
from optimizer import DeterministicQGAOptimizer, OptimizationInput


@dataclass(frozen=True)
class Settings:
    default_iterations: int
    log_level: str
    log_format: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            default_iterations=int(os.getenv("QGA_DEFAULT_ITERATIONS", "20")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "json"),
        )


class OptimizeRequest(BaseModel):
    available_features: int = Field(..., ge=1, le=10000)
    latency_budget_ms: float = Field(..., ge=0.0)
    energy_budget: float = Field(..., ge=0.0, le=1.0)
    risk_tolerance: float = Field(..., ge=0.0, le=1.0)


class OptimizeResponse(BaseModel):
    status: str
    selected_features: list[str]
    feature_budget: int
    threshold_suggestion: float
    optimization_score: float
    qga_iterations: int
    mode: str


def configure_logging(settings: Settings) -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level.upper(),
        serialize=settings.log_format.lower() == "json",
        backtrace=False,
        diagnose=False,
    )


settings = Settings.from_env()
configure_logging(settings)
optimizer = DeterministicQGAOptimizer(default_iterations=settings.default_iterations)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    QGA_SERVICE_STATUS.set(1)
    logger.info(
        "qga_service_startup",
        mode="deterministic_stub",
        default_iterations=settings.default_iterations,
    )
    try:
        yield
    finally:
        QGA_SERVICE_STATUS.set(0)
        logger.info("qga_service_shutdown")


app = FastAPI(
    title="QI-FL-IDS-IoT qga-service",
    version="p6c",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "qga-service",
        "mode": "deterministic_stub",
        "version": "p6c",
    }


@app.get("/ready")
def ready(response: Response) -> dict[str, Any]:
    optimizer_loaded = optimizer is not None
    if not optimizer_loaded:
        response.status_code = 503

    return {
        "status": "ready" if optimizer_loaded else "not_ready",
        "service": "qga-service",
        "ready": optimizer_loaded,
        "mode": "deterministic_stub",
        "optimizer_loaded": optimizer_loaded,
    }


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(payload: OptimizeRequest) -> OptimizeResponse:
    started = time.perf_counter()
    try:
        result = optimizer.optimize(
            OptimizationInput(
                available_features=payload.available_features,
                latency_budget_ms=payload.latency_budget_ms,
                energy_budget=payload.energy_budget,
                risk_tolerance=payload.risk_tolerance,
            )
        )
        QGA_LAST_SCORE.set(result.optimization_score)
        QGA_REQUESTS_TOTAL.labels(status="ok").inc()
        logger.info(
            "qga_optimization_completed",
            available_features=payload.available_features,
            feature_budget=result.feature_budget,
            optimization_score=result.optimization_score,
            qga_iterations=result.qga_iterations,
            mode=result.mode,
        )
        return OptimizeResponse(**result.__dict__)
    except Exception:
        QGA_REQUESTS_TOTAL.labels(status="error").inc()
        logger.exception("qga_optimization_failed")
        raise
    finally:
        QGA_OPTIMIZATION_LATENCY_SECONDS.observe(time.perf_counter() - started)


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=prometheus_text(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
