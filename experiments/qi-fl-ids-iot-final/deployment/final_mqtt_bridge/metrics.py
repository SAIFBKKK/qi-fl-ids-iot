from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from threading import Lock


HTTP_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


@dataclass
class BridgeMetrics:
    started_at: float = field(default_factory=time.time)
    ready: bool = False
    mqtt_connected: bool = False
    flows_received: Counter[tuple[str, str]] = field(default_factory=Counter)
    predictions_published: Counter[tuple[str, str]] = field(default_factory=Counter)
    alerts_published: Counter[tuple[str, str]] = field(default_factory=Counter)
    prediction_errors: Counter[str] = field(default_factory=Counter)
    latency_buckets: Counter[float] = field(default_factory=Counter)
    latency_count: int = 0
    latency_sum: float = 0.0
    last_message_timestamp_seconds: float = 0.0
    last_error: str | None = None
    _lock: Lock = field(default_factory=Lock, repr=False)

    def set_ready(self, value: bool) -> None:
        with self._lock:
            self.ready = value

    def set_mqtt_connected(self, value: bool) -> None:
        with self._lock:
            self.mqtt_connected = value

    def mark_flow_received(self, node_id: str, source_topic: str) -> None:
        with self._lock:
            self.flows_received[(node_id, source_topic)] += 1
            self.last_message_timestamp_seconds = time.time()

    def mark_prediction_published(self, node_id: str, predicted_label: str) -> None:
        with self._lock:
            self.predictions_published[(node_id, predicted_label)] += 1

    def mark_alert_published(self, node_id: str, severity: str) -> None:
        with self._lock:
            self.alerts_published[(node_id, severity)] += 1

    def mark_error(self, reason: str, error: str | None = None) -> None:
        with self._lock:
            self.prediction_errors[reason] += 1
            self.last_error = error or reason

    def observe_http_latency(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            return
        with self._lock:
            self.latency_count += 1
            self.latency_sum += seconds
            for bucket in HTTP_LATENCY_BUCKETS:
                if seconds <= bucket:
                    self.latency_buckets[bucket] += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "ready": self.ready,
                "mqtt_connected": self.mqtt_connected,
                "uptime_seconds": round(time.time() - self.started_at, 3),
                "flows_received_total": sum(self.flows_received.values()),
                "predictions_published_total": sum(self.predictions_published.values()),
                "alerts_published_total": sum(self.alerts_published.values()),
                "prediction_errors_total": sum(self.prediction_errors.values()),
                "last_message_timestamp_seconds": self.last_message_timestamp_seconds,
                "last_error": self.last_error,
            }

    def prometheus_text(self) -> str:
        with self._lock:
            lines = [
                "# HELP final_mqtt_bridge_ready Readiness state for final-mqtt-bridge.",
                "# TYPE final_mqtt_bridge_ready gauge",
                f"final_mqtt_bridge_ready {1 if self.ready else 0}",
                "# HELP final_mqtt_bridge_mqtt_connected MQTT connection state.",
                "# TYPE final_mqtt_bridge_mqtt_connected gauge",
                f"final_mqtt_bridge_mqtt_connected {1 if self.mqtt_connected else 0}",
                "# HELP final_mqtt_bridge_flows_received_total Replayed MQTT flows received.",
                "# TYPE final_mqtt_bridge_flows_received_total counter",
            ]
            if not self.flows_received:
                lines.append('final_mqtt_bridge_flows_received_total{node_id="none",source_topic="none"} 0')
            for (node_id, source_topic), value in sorted(self.flows_received.items()):
                lines.append(
                    "final_mqtt_bridge_flows_received_total"
                    f'{{node_id="{_escape(node_id)}",source_topic="{_escape(source_topic)}"}} {value}'
                )

            lines.extend(
                [
                    "# HELP final_mqtt_bridge_predictions_published_total Final IDS prediction messages published to MQTT.",
                    "# TYPE final_mqtt_bridge_predictions_published_total counter",
                ]
            )
            if not self.predictions_published:
                lines.append('final_mqtt_bridge_predictions_published_total{node_id="none",predicted_label="none"} 0')
            for (node_id, predicted_label), value in sorted(self.predictions_published.items()):
                lines.append(
                    "final_mqtt_bridge_predictions_published_total"
                    f'{{node_id="{_escape(node_id)}",predicted_label="{_escape(predicted_label)}"}} {value}'
                )

            lines.extend(
                [
                    "# HELP final_mqtt_bridge_alerts_published_total Final IDS alert messages published to MQTT.",
                    "# TYPE final_mqtt_bridge_alerts_published_total counter",
                ]
            )
            if not self.alerts_published:
                lines.append('final_mqtt_bridge_alerts_published_total{node_id="none",severity="none"} 0')
            for (node_id, severity), value in sorted(self.alerts_published.items()):
                lines.append(
                    "final_mqtt_bridge_alerts_published_total"
                    f'{{node_id="{_escape(node_id)}",severity="{_escape(severity)}"}} {value}'
                )

            lines.extend(
                [
                    "# HELP final_mqtt_bridge_prediction_errors_total Errors while converting MQTT flows into final IDS predictions.",
                    "# TYPE final_mqtt_bridge_prediction_errors_total counter",
                ]
            )
            if not self.prediction_errors:
                lines.append('final_mqtt_bridge_prediction_errors_total{reason="none"} 0')
            for reason, value in sorted(self.prediction_errors.items()):
                lines.append(f'final_mqtt_bridge_prediction_errors_total{{reason="{_escape(reason)}"}} {value}')

            lines.extend(
                [
                    "# HELP final_mqtt_bridge_http_latency_seconds HTTP latency to final-ids-api /predict.",
                    "# TYPE final_mqtt_bridge_http_latency_seconds histogram",
                ]
            )
            for bucket in HTTP_LATENCY_BUCKETS:
                lines.append(f'final_mqtt_bridge_http_latency_seconds_bucket{{le="{bucket}"}} {self.latency_buckets[bucket]}')
            lines.append(f'final_mqtt_bridge_http_latency_seconds_bucket{{le="+Inf"}} {self.latency_count}')
            lines.append(f"final_mqtt_bridge_http_latency_seconds_count {self.latency_count}")
            lines.append(f"final_mqtt_bridge_http_latency_seconds_sum {self.latency_sum:.9f}")

            lines.extend(
                [
                    "# HELP final_mqtt_bridge_last_message_timestamp_seconds Unix timestamp of the last MQTT flow message received.",
                    "# TYPE final_mqtt_bridge_last_message_timestamp_seconds gauge",
                    f"final_mqtt_bridge_last_message_timestamp_seconds {self.last_message_timestamp_seconds:.6f}",
                    "# HELP final_mqtt_bridge_uptime_seconds Process uptime seconds.",
                    "# TYPE final_mqtt_bridge_uptime_seconds gauge",
                    f"final_mqtt_bridge_uptime_seconds {time.time() - self.started_at:.3f}",
                    "",
                ]
            )
            return "\n".join(lines)
