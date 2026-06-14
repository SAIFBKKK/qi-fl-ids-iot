from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest


QGA_REQUESTS_TOTAL = Counter(
    "qga_requests_total",
    "Total optimization requests handled by qga-service.",
    ["status"],
)

QGA_OPTIMIZATION_LATENCY_SECONDS = Histogram(
    "qga_optimization_latency_seconds",
    "Latency of qga-service optimization requests in seconds.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

QGA_LAST_SCORE = Gauge(
    "qga_last_score",
    "Most recent deterministic optimization score returned by qga-service.",
)

QGA_SERVICE_STATUS = Gauge(
    "qga_service_status",
    "qga-service availability status, 1 for healthy and 0 for stopped.",
)


def prometheus_text() -> bytes:
    return generate_latest()
