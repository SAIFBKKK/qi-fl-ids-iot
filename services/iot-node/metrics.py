from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any

INFERENCE_LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
KNOWN_REJECTION_REASONS = (
    "invalid_json",
    "invalid_schema",
    "missing_feature",
    "unexpected_feature",
    "invalid_feature_value",
    "preprocessing_error",
    "inference_error",
)


@dataclass
class NodeMetrics:
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    mqtt_connected: bool = False
    node_up: bool = False
    last_message_at: str | None = None
    last_prediction_at: str | None = None
    last_error: str | None = None
    flows_received: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    flows_rejected: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    predictions: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    alerts: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    latency_buckets: dict[float, int] = field(default_factory=lambda: defaultdict(int))
    latency_count: int = 0
    latency_sum: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def set_node_status(self, up: bool) -> None:
        with self._lock:
            self.node_up = up

    def mark_mqtt_connected(self, connected: bool) -> None:
        with self._lock:
            self.mqtt_connected = connected

    def mark_received(self, source_topic: str) -> None:
        with self._lock:
            self.flows_received[source_topic] += 1
            self.last_message_at = datetime.now(UTC).isoformat()

    def mark_rejected(self, reason: str, error: str | None = None) -> None:
        with self._lock:
            self.flows_rejected[reason] += 1
            self.last_error = error or reason

    def mark_prediction(self, predicted_label: str) -> None:
        with self._lock:
            self.predictions[predicted_label] += 1
            self.last_prediction_at = datetime.now(UTC).isoformat()

    def mark_alert(self, severity: str, predicted_label: str) -> None:
        with self._lock:
            self.alerts[(severity, predicted_label)] += 1

    def observe_inference_latency(self, seconds: float) -> None:
        with self._lock:
            self.latency_count += 1
            self.latency_sum += seconds
            for bucket in INFERENCE_LATENCY_BUCKETS:
                if seconds <= bucket:
                    self.latency_buckets[bucket] += 1

    def mark_error(self, reason: str) -> None:
        with self._lock:
            self.last_error = reason

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            uptime_seconds = (datetime.now(UTC) - self.started_at).total_seconds()
            return {
                "started_at": self.started_at.isoformat(),
                "uptime_seconds": round(uptime_seconds, 3),
                "mqtt_connected": self.mqtt_connected,
                "node_up": self.node_up,
                "last_message_at": self.last_message_at,
                "last_prediction_at": self.last_prediction_at,
                "last_error": self.last_error,
                "flows_received_total": sum(self.flows_received.values()),
                "flows_rejected_total": sum(self.flows_rejected.values()),
                "predictions_total": sum(self.predictions.values()),
                "alerts_total": sum(self.alerts.values()),
            }

    def prometheus_text(self, node_id: str, default_source_topic: str | None = None) -> str:
        with self._lock:
            lines = [
                "# HELP ids_flows_received_total MQTT flow messages received.",
                "# TYPE ids_flows_received_total counter",
            ]
            received_topics = dict(self.flows_received)
            if default_source_topic is not None:
                received_topics.setdefault(default_source_topic, 0)
            for source_topic, value in sorted(received_topics.items()):
                lines.append(
                    "ids_flows_received_total"
                    f'{{node_id="{_escape(node_id)}",source_topic="{_escape(source_topic)}"}} {value}'
                )

            lines.extend(
                [
                    "# HELP ids_flows_rejected_invalid_schema_total Invalid flow messages rejected.",
                    "# TYPE ids_flows_rejected_invalid_schema_total counter",
                ]
            )
            rejected = dict(self.flows_rejected)
            for reason in KNOWN_REJECTION_REASONS:
                rejected.setdefault(reason, 0)
            for reason, value in sorted(rejected.items()):
                lines.append(
                    "ids_flows_rejected_invalid_schema_total"
                    f'{{node_id="{_escape(node_id)}",reason="{_escape(reason)}"}} {value}'
                )

            lines.extend(
                [
                    "# HELP ids_predictions_total Prediction messages published.",
                    "# TYPE ids_predictions_total counter",
                ]
            )
            for predicted_label, value in sorted(self.predictions.items()):
                lines.append(
                    "ids_predictions_total"
                    f'{{node_id="{_escape(node_id)}",predicted_label="{_escape(predicted_label)}"}} {value}'
                )

            lines.extend(
                [
                    "# HELP ids_alerts_total Alert messages published.",
                    "# TYPE ids_alerts_total counter",
                ]
            )
            for (severity, predicted_label), value in sorted(self.alerts.items()):
                lines.append(
                    "ids_alerts_total"
                    f'{{node_id="{_escape(node_id)}",severity="{_escape(severity)}",'
                    f'predicted_label="{_escape(predicted_label)}"}} {value}'
                )

            lines.extend(
                [
                    "# HELP inference_latency_seconds Inference processing latency.",
                    "# TYPE inference_latency_seconds histogram",
                ]
            )
            cumulative = 0
            for bucket in INFERENCE_LATENCY_BUCKETS:
                cumulative = self.latency_buckets[bucket]
                lines.append(
                    "inference_latency_seconds_bucket"
                    f'{{node_id="{_escape(node_id)}",le="{bucket}"}} {cumulative}'
                )
            lines.append(
                "inference_latency_seconds_bucket"
                f'{{node_id="{_escape(node_id)}",le="+Inf"}} {self.latency_count}'
            )
            lines.append(f'inference_latency_seconds_count{{node_id="{_escape(node_id)}"}} {self.latency_count}')
            lines.append(f'inference_latency_seconds_sum{{node_id="{_escape(node_id)}"}} {self.latency_sum:.9f}')

            lines.extend(
                [
                    "# HELP ids_node_status IoT node process status.",
                    "# TYPE ids_node_status gauge",
                    f'ids_node_status{{node_id="{_escape(node_id)}"}} {1 if self.node_up else 0}',
                    "",
                ]
            )
            return "\n".join(lines)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
