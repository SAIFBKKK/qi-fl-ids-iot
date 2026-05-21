from __future__ import annotations

import json
import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any


def topic_family(topic: str) -> str:
    if topic.startswith("ids/flows/"):
        return "flows"
    if topic.startswith("ids/predictions/"):
        return "predictions"
    if topic.startswith("ids/alerts/"):
        return "alerts"
    if topic.startswith("ids/status/"):
        return "status"
    return "other"


def parse_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


@dataclass
class OnlineValidatorMetrics:
    mqtt_connected: bool = False
    total_messages: int = 0
    topic_counts: Counter[str] = field(default_factory=Counter)
    family_counts: Counter[str] = field(default_factory=Counter)
    flow_first_seen: dict[str, float] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)
    samples: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=25))
    last_error: str | None = None
    started_at: float = field(default_factory=time.time)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def set_connected(self, connected: bool) -> None:
        with self._lock:
            self.mqtt_connected = connected

    def mark_error(self, error: str) -> None:
        with self._lock:
            self.last_error = error

    def record_message(self, topic: str, payload_bytes: bytes) -> None:
        now = time.time()
        payload_text = payload_bytes.decode("utf-8", errors="replace")
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            payload = {"raw": payload_text[:500]}

        family = topic_family(topic)
        flow_id = payload.get("flow_id") if isinstance(payload, dict) else None
        timestamp = parse_timestamp(payload.get("timestamp")) if isinstance(payload, dict) else None

        with self._lock:
            self.total_messages += 1
            self.topic_counts[topic] += 1
            self.family_counts[family] += 1
            if family == "flows" and flow_id:
                self.flow_first_seen[str(flow_id)] = timestamp or now
            elif family in {"predictions", "alerts"} and flow_id:
                started = self.flow_first_seen.get(str(flow_id))
                if started is not None:
                    ended = timestamp or now
                    latency_ms = max((ended - started) * 1000.0, 0.0)
                    if math.isfinite(latency_ms):
                        self.latencies_ms.append(latency_ms)
            self.samples.append(
                {
                    "topic": topic,
                    "family": family,
                    "flow_id": flow_id,
                    "timestamp": payload.get("timestamp") if isinstance(payload, dict) else None,
                    "received_at_unix": now,
                    "payload_preview": payload_text[:250],
                }
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencies = list(self.latencies_ms)
            return {
                "mqtt_connected": self.mqtt_connected,
                "uptime_seconds": round(time.time() - self.started_at, 3),
                "total_messages": self.total_messages,
                "topic_counts": dict(self.topic_counts),
                "family_counts": dict(self.family_counts),
                "latency_observations": len(latencies),
                "latency_mean_ms": sum(latencies) / len(latencies) if latencies else None,
                "latency_max_ms": max(latencies) if latencies else None,
                "samples": list(self.samples),
                "last_error": self.last_error,
            }

    def prometheus_text(self) -> str:
        with self._lock:
            lines = [
                "# HELP online_validator_mqtt_connected MQTT connection state.",
                "# TYPE online_validator_mqtt_connected gauge",
                f"online_validator_mqtt_connected {1 if self.mqtt_connected else 0}",
                "# HELP online_validator_messages_total Observed MQTT messages.",
                "# TYPE online_validator_messages_total counter",
            ]
            for family, value in sorted(self.family_counts.items()):
                lines.append(f'online_validator_messages_total{{family="{escape_label(family)}"}} {value}')

            lines.extend(
                [
                    "# HELP online_validator_topic_messages_total Observed MQTT messages by topic.",
                    "# TYPE online_validator_topic_messages_total counter",
                ]
            )
            for topic, value in sorted(self.topic_counts.items()):
                lines.append(f'online_validator_topic_messages_total{{topic="{escape_label(topic)}"}} {value}')

            count = len(self.latencies_ms)
            latency_sum = sum(self.latencies_ms) / 1000.0
            lines.extend(
                [
                    "# HELP online_validator_flow_to_prediction_latency_seconds Flow-to-prediction latency observations.",
                    "# TYPE online_validator_flow_to_prediction_latency_seconds summary",
                    f'online_validator_flow_to_prediction_latency_seconds_count {count}',
                    f'online_validator_flow_to_prediction_latency_seconds_sum {latency_sum}',
                    "",
                ]
            )
            return "\n".join(lines)
