"""
feature-extractor — agrège des événements IoT bruts par fenêtre de 60 s,
extrait 28 features CIC-IoT et publie le vecteur scalé sur MQTT.

Topics MQTT :
  input  : iot/raw_events     (JSON RawEvent)
  output : iot/features       (JSON FeatureVector)
HTTP :
  GET /health  → 200 {"status": "ok", ...}
  GET /metrics → 200 texte Prometheus
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import paho.mqtt.client as mqtt

from feature_map import (
    FEATURE_NAMES,
    build_feature_vector,
)

# ---------------------------------------------------------------------------
# Logging JSON structuré
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    def __init__(self, node_id: str) -> None:
        super().__init__()
        self._node_id = node_id

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "message": record.getMessage(),
            "node_id": self._node_id,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(node_id: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("feature-extractor")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter(node_id))
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Modèles de données
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RawEvent:
    timestamp: float
    src_ip: str
    dst_ip: str
    protocol: str
    src_port: int
    dst_port: int
    pkt_count: int
    byte_count: int
    direction: str
    # v2 — champs optionnels pour couverture complète des 28 features
    header_length: int = 20    # octets d'en-tête (défaut TCP standard)
    fin_flag: int = 0
    syn_flag: int = 0
    rst_flag: int = 0
    psh_flag: int = 0
    ack_flag: int = 0
    urg_flag: int = 0          # quasi-absent dans CIC-IoT-2023

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RawEvent":
        return cls(
            timestamp=float(d["timestamp"]),
            src_ip=str(d["src_ip"]),
            dst_ip=str(d["dst_ip"]),
            protocol=str(d.get("protocol", "TCP")).upper(),
            src_port=int(d.get("src_port", 0)),
            dst_port=int(d.get("dst_port", 0)),
            pkt_count=int(d.get("pkt_count", 1)),
            byte_count=int(d.get("byte_count", 0)),
            direction=str(d.get("direction", "outbound")).lower(),
            header_length=int(d.get("header_length", 20)),
            fin_flag=int(d.get("fin_flag", 0)),
            syn_flag=int(d.get("syn_flag", 0)),
            rst_flag=int(d.get("rst_flag", 0)),
            psh_flag=int(d.get("psh_flag", 0)),
            ack_flag=int(d.get("ack_flag", 0)),
            urg_flag=int(d.get("urg_flag", 0)),
        )


@dataclass
class WindowBuffer:
    window_start: float
    events: list[RawEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agrégation par fenêtre
# ---------------------------------------------------------------------------

class WindowAggregator:
    """Accumule les événements bruts par src_ip dans des fenêtres temporelles.

    Thread-safe : add_event() et flush_expired() peuvent être appelés depuis
    des threads différents.
    """

    def __init__(self, window_size_s: float) -> None:
        if window_size_s <= 0:
            raise ValueError("window_size_s must be > 0")
        self.window_size_s = window_size_s
        self._windows: dict[str, WindowBuffer] = {}
        self._lock = threading.Lock()

    def add_event(self, event: RawEvent) -> None:
        with self._lock:
            buf = self._windows.get(event.src_ip)
            if buf is None:
                self._windows[event.src_ip] = WindowBuffer(
                    window_start=event.timestamp,
                    events=[event],
                )
            else:
                buf.events.append(event)

    def flush_expired(self, now: float) -> list[tuple[str, list[RawEvent]]]:
        """Retourne et supprime les fenêtres dont l'âge >= window_size_s."""
        expired: list[tuple[str, list[RawEvent]]] = []
        with self._lock:
            for src_ip, buf in list(self._windows.items()):
                if now - buf.window_start >= self.window_size_s:
                    expired.append((src_ip, list(buf.events)))
                    del self._windows[src_ip]
        return expired

    def active_window_count(self) -> int:
        with self._lock:
            return len(self._windows)


# ---------------------------------------------------------------------------
# Construction du dict de fenêtre pour build_feature_vector()
# ---------------------------------------------------------------------------

def build_window_dict(events: list[RawEvent]) -> dict[str, Any]:
    """Transforme une liste de RawEvent en dict de fenêtre attendu par feature_map.

    Lève ValueError si events est vide.
    """
    if not events:
        raise ValueError("events list is empty — cannot build window dict")

    from collections import Counter
    timestamps = [e.timestamp for e in events]
    flow_duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0

    proto_counter  = Counter(e.protocol for e in events)
    dominant_proto = proto_counter.most_common(1)[0][0]
    mean_header    = sum(e.header_length for e in events) / len(events)

    return {
        "flow_duration": flow_duration,
        "header_length": mean_header,
        "protocol":      dominant_proto,
        "pkt_bytes":     [e.byte_count for e in events],
        "timestamps":    timestamps,
        "dst_ports":     [e.dst_port   for e in events],
        "protocols":     [e.protocol   for e in events],
        "fin_flags":     [e.fin_flag   for e in events],
        "syn_flags":     [e.syn_flag   for e in events],
        "rst_flags":     [e.rst_flag   for e in events],
        "psh_flags":     [e.psh_flag   for e in events],
        "ack_flags":     [e.ack_flag   for e in events],
        "urg_flags":     [e.urg_flag   for e in events],
    }


# ---------------------------------------------------------------------------
# Chargement des artefacts
# ---------------------------------------------------------------------------

def load_artifacts(
    scaler_path: str,
    feature_names_path: str,
) -> tuple[Any, list[str]]:
    """Charge scaler et feature_names depuis le volume artifacts.

    Supporte feature_names.pkl (joblib) et feature_names.json (liste).
    Valide que l'ordre du bundle correspond à FEATURE_NAMES de feature_map.py.
    Retourne (scaler, feature_names).
    """
    scaler = joblib.load(scaler_path)

    fn_path = Path(feature_names_path)
    if fn_path.suffix == ".json":
        with open(fn_path, encoding="utf-8") as f:
            feature_names = json.load(f)
        if not isinstance(feature_names, list):
            raise ValueError("feature_names.json must contain a JSON array of strings")
    else:
        feature_names = list(joblib.load(fn_path))

    if len(feature_names) != 28:
        raise ValueError(f"feature_names must contain 28 entries, found {len(feature_names)}")
    if feature_names != FEATURE_NAMES:
        raise ValueError(
            "feature_names loaded from disk does not match FEATURE_NAMES — "
            "verify the bundle and update feature_map.py if needed"
        )

    return scaler, feature_names


def apply_scaler(scaler: Any, vector: list[float]) -> list[float]:
    """Applique scaler.transform sur le vecteur 1×28 et retourne une liste de 28 floats."""
    arr = np.asarray([vector], dtype=np.float32)
    scaled = np.asarray(scaler.transform(arr), dtype=np.float64)
    if not np.isfinite(scaled).all():
        raise ValueError("scaler produced non-finite values — check input features")
    return list(scaled[0])


# ---------------------------------------------------------------------------
# Endpoints HTTP /health et /metrics
# ---------------------------------------------------------------------------

def _prometheus_text(state: dict[str, Any]) -> str:
    node_id = state.get("node_id", "unknown")
    label = f'node_id="{node_id}"'
    lines = [
        "# HELP feature_extractor_status feature-extractor process status.",
        "# TYPE feature_extractor_status gauge",
        f"feature_extractor_status{{{label}}} {1 if state.get('mqtt_connected', False) else 0}",
        "# HELP feature_extractor_mqtt_connected MQTT connection state.",
        "# TYPE feature_extractor_mqtt_connected gauge",
        f"feature_extractor_mqtt_connected{{{label}}} {1 if state.get('mqtt_connected', False) else 0}",
        "# HELP feature_extractor_windows_active Active time-windows in aggregator.",
        "# TYPE feature_extractor_windows_active gauge",
        f"feature_extractor_windows_active{{{label}}} {state.get('windows_active', 0)}",
        "# HELP feature_extractor_events_received_total Raw events received from MQTT.",
        "# TYPE feature_extractor_events_received_total counter",
        f"feature_extractor_events_received_total{{{label}}} {state.get('events_received', 0)}",
        "# HELP feature_extractor_events_rejected_total Raw events rejected (invalid).",
        "# TYPE feature_extractor_events_rejected_total counter",
        f"feature_extractor_events_rejected_total{{{label}}} {state.get('events_rejected', 0)}",
        "# HELP feature_extractor_vectors_published_total Feature vectors published to MQTT.",
        "# TYPE feature_extractor_vectors_published_total counter",
        f"feature_extractor_vectors_published_total{{{label}}} {state.get('vectors_published', 0)}",
        "",
    ]
    return "\n".join(lines)


class _HealthHandler(BaseHTTPRequestHandler):
    _state: dict[str, Any] = {}

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            body = json.dumps({
                "status": "ok",
                "service": "feature-extractor",
                "node_id": self._state.get("node_id", "unknown"),
                "mqtt_connected": self._state.get("mqtt_connected", False),
                "windows_active": self._state.get("windows_active", 0),
                "vectors_published": self._state.get("vectors_published", 0),
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/metrics":
            body = _prometheus_text(self._state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:  # silence access log
        pass


def _start_health_server(port: int, state: dict[str, Any]) -> None:
    _HealthHandler._state = state
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Service principal
# ---------------------------------------------------------------------------

class FeatureExtractorService:
    def __init__(
        self,
        node_id: str,
        broker: str,
        port: int,
        window_size_s: float,
        scaler: Any,
        feature_names: list[str],
        input_topic: str = "iot/raw_events",
        output_topic: str = "iot/features",
        username: str | None = None,
        password: str | None = None,
        health_port: int = 8000,
        flush_interval_s: float = 1.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.node_id        = node_id
        self.broker         = broker
        self.port           = port
        self.window_size_s  = window_size_s
        self.scaler         = scaler
        self.feature_names  = feature_names
        self.input_topic    = input_topic
        self.output_topic   = output_topic
        self.flush_interval_s = flush_interval_s
        self.log = logger or logging.getLogger("feature-extractor")

        self._aggregator = WindowAggregator(window_size_s)
        self._state: dict[str, Any] = {
            "node_id": node_id,
            "mqtt_connected": False,
            "windows_active": 0,
            "events_received": 0,
            "events_rejected": 0,
            "vectors_published": 0,
        }
        self._client = self._build_client(username, password)
        _start_health_server(health_port, self._state)

    # ------------------------------------------------------------------
    # MQTT
    # ------------------------------------------------------------------

    def _build_client(self, username: str | None, password: str | None) -> mqtt.Client:
        client_id = f"feature-extractor-{self.node_id}"
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        except (AttributeError, TypeError):
            client = mqtt.Client(client_id=client_id)

        if username:
            client.username_pw_set(username, password)

        client.reconnect_delay_set(min_delay=1, max_delay=30)
        client.on_connect    = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message    = self._on_message
        return client

    def _on_connect(self, client: mqtt.Client, _ud: Any, _flags: Any, rc: Any, *_: Any) -> None:
        rc_val = getattr(rc, "value", rc)
        try:
            rc_int = int(rc_val)
        except (TypeError, ValueError):
            rc_int = 0 if str(rc_val).lower() == "success" else 1

        if rc_int != 0:
            self.log.error("mqtt_connect_failed", extra={"rc": str(rc)})
            self._state["mqtt_connected"] = False
            return

        self._state["mqtt_connected"] = True
        client.subscribe(self.input_topic, qos=1)
        self.log.info(
            "mqtt_connected broker=%s port=%d subscribed=%s",
            self.broker, self.port, self.input_topic,
        )

    def _on_disconnect(self, _client: mqtt.Client, _ud: Any, *args: Any) -> None:
        self._state["mqtt_connected"] = False
        self.log.warning("mqtt_disconnected")

    def _on_message(self, _client: mqtt.Client, _ud: Any, message: mqtt.MQTTMessage) -> None:
        self._state["events_received"] = self._state.get("events_received", 0) + 1
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.log.warning("invalid_json topic=%s err=%s", message.topic, exc)
            self._state["events_rejected"] = self._state.get("events_rejected", 0) + 1
            return

        try:
            event = RawEvent.from_dict(payload)
        except (KeyError, ValueError, TypeError) as exc:
            self.log.warning("invalid_event err=%s", exc)
            self._state["events_rejected"] = self._state.get("events_rejected", 0) + 1
            return

        self._aggregator.add_event(event)
        self._state["windows_active"] = self._aggregator.active_window_count()

    # ------------------------------------------------------------------
    # Flush loop
    # ------------------------------------------------------------------

    def _flush_loop(self) -> None:
        while True:
            time.sleep(self.flush_interval_s)
            self._process_expired_windows()

    def _process_expired_windows(self) -> None:
        now = time.time()
        for src_ip, events in self._aggregator.flush_expired(now):
            try:
                self._publish_features(src_ip, events, now)
            except Exception as exc:  # noqa: BLE001 - never crash the flush loop
                self.log.error("flush_error src_ip=%s err=%s", src_ip, exc)
        self._state["windows_active"] = self._aggregator.active_window_count()

    def _publish_features(self, src_ip: str, events: list[RawEvent], now: float) -> None:
        window     = build_window_dict(events)
        raw_vector = build_feature_vector(window)
        scaled     = apply_scaler(self.scaler, raw_vector)

        message = {
            "schema_version": "1.0",
            "event_type":     "feature_vector",
            "node_id":        self.node_id,
            "src_ip":         src_ip,
            "timestamp":      datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "window_size_s":  self.window_size_s,
            "event_count":    len(events),
            "feature_names":  self.feature_names,
            "vector":         scaled,
        }
        result = self._client.publish(
            self.output_topic,
            json.dumps(message, separators=(",", ":")),
            qos=1,
        )
        rc = getattr(result, "rc", result)
        if int(rc) != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish failed rc={rc}")

        self._state["vectors_published"] = self._state.get("vectors_published", 0) + 1
        self.log.info(
            "vector_published src_ip=%s events=%d topic=%s",
            src_ip, len(events), self.output_topic,
        )

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.log.info(
            "starting node_id=%s broker=%s port=%d window=%ss",
            self.node_id, self.broker, self.port, self.window_size_s,
        )
        self._client.connect_async(self.broker, self.port, keepalive=30)
        self._client.loop_start()
        flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        flush_thread.start()

    def stop(self) -> None:
        self.log.info("stopping node_id=%s", self.node_id)
        self._client.loop_stop()
        self._client.disconnect()


# ---------------------------------------------------------------------------
# Entrée principale
# ---------------------------------------------------------------------------

def main() -> None:
    node_id       = os.environ.get("NODE_ID",       "extractor-1")
    broker        = os.environ.get("MQTT_BROKER",   "localhost")
    port          = int(os.environ.get("MQTT_PORT",  "1883"))
    username      = os.environ.get("MQTT_USERNAME")  or None
    password      = os.environ.get("MQTT_PASSWORD")  or None
    window_size_s = float(os.environ.get("WINDOW_SIZE_S", "60"))
    scaler_path   = os.environ.get("SCALER_PATH",        "/artifacts/scaler.pkl")
    feature_names_path = os.environ.get(
        "FEATURE_NAMES_PATH", "/artifacts/feature_names.pkl"
    )
    input_topic   = os.environ.get("INPUT_TOPIC",  "iot/raw_events")
    output_topic  = os.environ.get("OUTPUT_TOPIC", "iot/features")
    health_port   = int(os.environ.get("HEALTH_PORT", "8000"))
    log_level     = os.environ.get("LOG_LEVEL", "INFO")

    logger = setup_logging(node_id, log_level)

    try:
        scaler, feature_names = load_artifacts(scaler_path, feature_names_path)
    except Exception as exc:
        logger.critical("artifact_load_failed err=%s", exc)
        sys.exit(1)

    service = FeatureExtractorService(
        node_id=node_id,
        broker=broker,
        port=port,
        window_size_s=window_size_s,
        scaler=scaler,
        feature_names=feature_names,
        input_topic=input_topic,
        output_topic=output_topic,
        username=username,
        password=password,
        health_port=health_port,
        logger=logger,
    )
    service.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()


if __name__ == "__main__":
    main()
