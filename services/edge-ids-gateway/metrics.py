from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest


EDGE_GATEWAY_REQUESTS_TOTAL = Counter(
    "edge_gateway_requests_total",
    "Total HTTP or processing requests seen by edge-ids-gateway.",
)

EDGE_GATEWAY_ALLOWED_TOTAL = Counter(
    "edge_gateway_allowed_total",
    "Total events allowed by edge-ids-gateway.",
)

EDGE_GATEWAY_BLOCKED_TOTAL = Counter(
    "edge_gateway_blocked_total",
    "Total events blocked by edge-ids-gateway.",
)

EDGE_GATEWAY_PREDICTIONS_TOTAL = Counter(
    "edge_gateway_predictions_total",
    "Total predictions produced by edge-ids-gateway.",
    ["predicted_label"],
)

EDGE_GATEWAY_ALERTS_TOTAL = Counter(
    "edge_gateway_alerts_total",
    "Total alerts produced by edge-ids-gateway.",
    ["severity"],
)

EDGE_GATEWAY_REJECTED_TOTAL = Counter(
    "edge_gateway_rejected_total",
    "Total events rejected by edge-ids-gateway.",
    ["reason"],
)

EDGE_GATEWAY_INFERENCE_LATENCY_SECONDS = Histogram(
    "edge_gateway_inference_latency_seconds",
    "Inference latency observed by edge-ids-gateway in seconds.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

EDGE_GATEWAY_STATUS = Gauge(
    "edge_gateway_status",
    "edge-ids-gateway availability status, 1 for up and 0 for down.",
)

EDGE_GATEWAY_READY = Gauge(
    "edge_gateway_ready",
    "edge-ids-gateway readiness status, 1 for ready and 0 for not ready.",
)

EDGE_GATEWAY_INFERENCE_READY = Gauge(
    "edge_gateway_inference_ready",
    "Inference readiness flag for edge-ids-gateway.",
)

EDGE_GATEWAY_MQTT_CONNECTED = Gauge(
    "edge_gateway_mqtt_connected",
    "MQTT connectivity flag for edge-ids-gateway.",
)

EDGE_GATEWAY_MODEL_READY = Gauge(
    "edge_gateway_model_ready",
    "Model readiness flag for edge-ids-gateway.",
)

EDGE_GATEWAY_ARTIFACT_MISSING_TOTAL = Counter(
    "edge_gateway_artifact_missing_total",
    "Number of missing bundle artifacts detected by edge-ids-gateway.",
    ["artifact"],
)

EDGE_GATEWAY_MQTT_MESSAGES_TOTAL = Counter(
    "edge_gateway_mqtt_messages_total",
    "Total MQTT messages received by edge-ids-gateway.",
    ["topic"],
)

EDGE_GATEWAY_STATUS.set(1)
EDGE_GATEWAY_READY.set(0)
EDGE_GATEWAY_INFERENCE_READY.set(0)
EDGE_GATEWAY_MQTT_CONNECTED.set(0)
EDGE_GATEWAY_MODEL_READY.set(0)


def prometheus_text() -> bytes:
    return generate_latest()
