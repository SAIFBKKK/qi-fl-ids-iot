from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class FLServerMetrics:
    """Métriques Prometheus du serveur FL — format texte custom, sans prometheus_client."""

    _current_round: int = 0
    _round_accuracy: float = 0.0
    _benign_recall: float = 0.0
    _f1_macro: float = 0.0
    _round_duration_seconds: float = 0.0
    _active_clients: int = 0
    _bandwidth_bytes: int = 0
    _false_positive_rate: float = 0.0
    _registered_nodes_total: int = 0
    _registered_nodes_by_tier: dict[str, int] = field(
        default_factory=lambda: {"weak": 0, "medium": 0, "powerful": 0}
    )
    _alerts: dict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _training_thread_status: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def update(
        self,
        round_num: int,
        accuracy: float,
        benign_recall: float,
        f1_macro: float,
        duration: float,
        active_clients: int,
        bandwidth_bytes: int,
        false_positive_rate: float = 0.0,
        alerts: dict[tuple[str, str], int] | None = None,
    ) -> None:
        """Met à jour toutes les métriques après un round FL.

        alerts : dict {(node, attack_type): count} — incrémente les compteurs.
        """
        with self._lock:
            self._current_round = round_num
            self._round_accuracy = accuracy
            self._benign_recall = benign_recall
            self._f1_macro = f1_macro
            self._round_duration_seconds = duration
            self._active_clients = active_clients
            self._bandwidth_bytes = bandwidth_bytes
            self._false_positive_rate = false_positive_rate
            if alerts:
                for (node, attack_type), count in alerts.items():
                    self._alerts[(node, attack_type)] += count

    def set_training_thread_status(self, status: int) -> None:
        with self._lock:
            self._training_thread_status = status

    def record_alert(self, node: str, attack_type: str, count: int = 1) -> None:
        with self._lock:
            self._alerts[(node, attack_type)] += count

    def update_registered_nodes(self, total: int, by_tier: dict[str, int]) -> None:
        with self._lock:
            self._registered_nodes_total = int(total)
            self._registered_nodes_by_tier = {
                "weak": int(by_tier.get("weak", 0)),
                "medium": int(by_tier.get("medium", 0)),
                "powerful": int(by_tier.get("powerful", 0)),
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "current_round": self._current_round,
                "round_accuracy": self._round_accuracy,
                "benign_recall": self._benign_recall,
                "f1_macro": self._f1_macro,
                "round_duration_seconds": self._round_duration_seconds,
                "active_clients": self._active_clients,
                "bandwidth_bytes": self._bandwidth_bytes,
                "false_positive_rate": self._false_positive_rate,
                "registered_nodes_total": self._registered_nodes_total,
                "registered_nodes_by_tier": dict(self._registered_nodes_by_tier),
                "alerts_total": sum(self._alerts.values()),
                "training_thread_status": self._training_thread_status,
            }

    def prometheus_text(self) -> str:
        with self._lock:
            lines = [
                "# HELP fl_current_round Round FL actuel.",
                "# TYPE fl_current_round gauge",
                f"fl_current_round {self._current_round}",

                "# HELP fl_round_accuracy Accuracy globale du modèle agrégé.",
                "# TYPE fl_round_accuracy gauge",
                f"fl_round_accuracy {self._round_accuracy:.6f}",

                "# HELP fl_benign_recall Benign Recall — KPI principal IDS.",
                "# TYPE fl_benign_recall gauge",
                f"fl_benign_recall {self._benign_recall:.6f}",

                "# HELP fl_f1_macro F1-Macro score global.",
                "# TYPE fl_f1_macro gauge",
                f"fl_f1_macro {self._f1_macro:.6f}",

                "# HELP fl_round_duration_seconds Durée du dernier round en secondes.",
                "# TYPE fl_round_duration_seconds gauge",
                f"fl_round_duration_seconds {self._round_duration_seconds:.3f}",

                "# HELP fl_active_clients Nombre de clients actifs ce round.",
                "# TYPE fl_active_clients gauge",
                f"fl_active_clients {self._active_clients}",

                "# HELP fl_bandwidth_bytes Bande passante totale (bytes) échangée ce round.",
                "# TYPE fl_bandwidth_bytes gauge",
                f"fl_bandwidth_bytes {self._bandwidth_bytes}",

                "# HELP ids_false_positive_rate Taux de faux positifs global.",
                "# TYPE ids_false_positive_rate gauge",
                f"ids_false_positive_rate {self._false_positive_rate:.6f}",

                "# HELP registered_nodes_total Total registered dynamic Mode A nodes.",
                "# TYPE registered_nodes_total gauge",
                f"registered_nodes_total {self._registered_nodes_total}",

                "# HELP registered_nodes_by_tier Registered dynamic Mode A nodes by assigned tier.",
                "# TYPE registered_nodes_by_tier gauge",
            ]
            for tier, value in sorted(self._registered_nodes_by_tier.items()):
                lines.append(f'registered_nodes_by_tier{{tier="{_escape(tier)}"}} {value}')

            lines.extend(
                [
                "# HELP ids_alerts_total Alertes IDS détectées par nœud et type d'attaque.",
                "# TYPE ids_alerts_total counter",
                ]
            )
            alerts = dict(self._alerts)
            if not alerts:
                alerts[("unknown", "unknown")] = 0
            for (node, attack_type), value in sorted(alerts.items()):
                lines.append(
                    "ids_alerts_total"
                    f'{{node="{_escape(node)}",attack_type="{_escape(attack_type)}"}} {value}'
                )

            lines.extend([
                "# HELP fl_training_thread_status Background training thread health (1=up, 0=down).",
                "# TYPE fl_training_thread_status gauge",
                f"fl_training_thread_status {self._training_thread_status}",
                "",
            ])
            return "\n".join(lines)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
