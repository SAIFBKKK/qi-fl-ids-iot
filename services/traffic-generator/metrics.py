from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class TrafficGeneratorMetrics:
    published_flows: int = 0
    skipped_rows: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    mqtt_connected: bool = False
    service_up: bool = False
    last_error: str | None = None
    _lock: Lock = field(default_factory=Lock, repr=False)

    def set_service_up(self, up: bool) -> None:
        with self._lock:
            self.service_up = up

    def set_mqtt_connected(self, connected: bool) -> None:
        with self._lock:
            self.mqtt_connected = connected

    def mark_published(self) -> None:
        with self._lock:
            self.published_flows += 1

    def mark_skipped(self, reason: str, error: str | None = None) -> None:
        with self._lock:
            self.skipped_rows[reason] += 1
            self.last_error = error or reason

    def mark_error(self, error: str) -> None:
        with self._lock:
            self.last_error = error

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "published_flows": self.published_flows,
                "skipped_rows": dict(self.skipped_rows),
                "mqtt_connected": self.mqtt_connected,
                "service_up": self.service_up,
                "last_error": self.last_error,
            }

    def prometheus_text(self, node_id: str, scenario: str) -> str:
        with self._lock:
            labels = f'node_id="{_escape(node_id)}",scenario="{_escape(scenario)}"'
            lines = [
                "# HELP traffic_generator_status Traffic generator process status.",
                "# TYPE traffic_generator_status gauge",
                f"traffic_generator_status{{{labels}}} {1 if self.service_up else 0}",
                "# HELP traffic_generator_flows_published_total Flow messages published.",
                "# TYPE traffic_generator_flows_published_total counter",
                f"traffic_generator_flows_published_total{{{labels}}} {self.published_flows}",
                "# HELP traffic_generator_rows_skipped_total Dataset rows skipped before publish.",
                "# TYPE traffic_generator_rows_skipped_total counter",
            ]

            skipped = dict(self.skipped_rows)
            skipped.setdefault("invalid_feature_value", 0)
            skipped.setdefault("publish_error", 0)
            for reason, value in sorted(skipped.items()):
                lines.append(
                    "traffic_generator_rows_skipped_total"
                    f'{{{labels},reason="{_escape(reason)}"}} {value}'
                )

            lines.extend(
                [
                    "# HELP traffic_generator_mqtt_connected MQTT connection state.",
                    "# TYPE traffic_generator_mqtt_connected gauge",
                    f'traffic_generator_mqtt_connected{{node_id="{_escape(node_id)}"}} '
                    f"{1 if self.mqtt_connected else 0}",
                    "",
                ]
            )
            return "\n".join(lines)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
