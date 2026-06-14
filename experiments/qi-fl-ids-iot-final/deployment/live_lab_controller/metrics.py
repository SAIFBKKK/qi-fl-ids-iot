from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from threading import Lock


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


@dataclass
class ControllerMetrics:
    started_at: float = field(default_factory=time.time)
    registrations: Counter[str] = field(default_factory=Counter)
    last_registration_timestamp_seconds: float = 0.0
    last_error: str | None = None
    _lock: Lock = field(default_factory=Lock, repr=False)

    def mark_registration(self, tier: str) -> None:
        with self._lock:
            self.registrations[tier] += 1
            self.last_registration_timestamp_seconds = time.time()

    def mark_error(self, error: str) -> None:
        with self._lock:
            self.last_error = error

    def snapshot(self, node_count: int) -> dict[str, object]:
        with self._lock:
            return {
                "uptime_seconds": round(time.time() - self.started_at, 3),
                "registered_nodes": node_count,
                "registrations_total": sum(self.registrations.values()),
                "last_registration_timestamp_seconds": self.last_registration_timestamp_seconds,
                "last_error": self.last_error,
            }

    def prometheus_text(self, node_count: int) -> str:
        with self._lock:
            lines = [
                "# HELP live_lab_controller_ready Readiness state for the live lab controller.",
                "# TYPE live_lab_controller_ready gauge",
                "live_lab_controller_ready 1",
                "# HELP live_lab_controller_registered_nodes Current registered IoT node count.",
                "# TYPE live_lab_controller_registered_nodes gauge",
                f"live_lab_controller_registered_nodes {node_count}",
                "# HELP live_lab_controller_registrations_total Node registration calls by assigned tier.",
                "# TYPE live_lab_controller_registrations_total counter",
            ]
            if not self.registrations:
                lines.append('live_lab_controller_registrations_total{tier="none"} 0')
            for tier, value in sorted(self.registrations.items()):
                lines.append(f'live_lab_controller_registrations_total{{tier="{_escape(tier)}"}} {value}')
            lines.extend(
                [
                    "# HELP live_lab_controller_last_registration_timestamp_seconds Unix timestamp of the last registration.",
                    "# TYPE live_lab_controller_last_registration_timestamp_seconds gauge",
                    f"live_lab_controller_last_registration_timestamp_seconds {self.last_registration_timestamp_seconds:.6f}",
                    "# HELP live_lab_controller_uptime_seconds Process uptime seconds.",
                    "# TYPE live_lab_controller_uptime_seconds gauge",
                    f"live_lab_controller_uptime_seconds {time.time() - self.started_at:.3f}",
                    "",
                ]
            )
            return "\n".join(lines)

