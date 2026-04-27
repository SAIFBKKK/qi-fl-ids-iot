from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import paho.mqtt.client as mqtt
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from loguru import logger

from metrics import TrafficGeneratorMetrics


@dataclass(frozen=True)
class Settings:
    node_id: str
    mqtt_broker: str
    mqtt_port: int
    mqtt_username: str | None
    mqtt_password: str | None
    replay_scenario: str
    replay_rate: float
    dataset_dir: Path
    artifacts_dir: Path
    log_level: str
    log_format: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        settings = cls(
            node_id=os.getenv("NODE_ID", "node1"),
            mqtt_broker=os.getenv("MQTT_BROKER", "mosquitto"),
            mqtt_port=int(os.getenv("MQTT_PORT", "1883")),
            mqtt_username=os.getenv("MQTT_USERNAME", "ids_user"),
            mqtt_password=os.getenv("MQTT_PASSWORD"),
            replay_scenario=os.getenv("REPLAY_SCENARIO", "mixed_chaos"),
            replay_rate=float(os.getenv("REPLAY_RATE", "5")),
            dataset_dir=Path(os.getenv("DATASET_DIR", "/data/demo")),
            artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", "/artifacts")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "json"),
        )
        if settings.replay_rate <= 0:
            fail_startup(f"REPLAY_RATE must be > 0, got {settings.replay_rate}")
        return settings


class TrafficReplayService:
    def __init__(self, settings: Settings, metrics: TrafficGeneratorMetrics) -> None:
        self.settings = settings
        self.metrics = metrics
        self.feature_names = self._load_feature_names()
        self.dataset_path = self.settings.dataset_dir / f"{self.settings.replay_scenario}.parquet"
        self.dataset = self._load_dataset()
        self.flow_topic = f"ids/flows/{self.settings.node_id}"
        self.status_topic = f"ids/status/{self.settings.node_id}"
        self.stop_event = threading.Event()
        self.connected_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.sequence = 0
        self.client = self._build_client()

    @property
    def rows_loaded(self) -> int:
        return len(self.dataset)

    def start(self) -> None:
        logger.info(
            "traffic_generator_starting",
            node_id=self.settings.node_id,
            scenario=self.settings.replay_scenario,
            rows_loaded=self.rows_loaded,
            replay_rate=self.settings.replay_rate,
        )
        self.stop_event.clear()
        self._connect_with_retry()
        self.metrics.set_service_up(True)
        self._publish_status("online")
        self.worker = threading.Thread(target=self._replay_loop, name="traffic-replay-loop", daemon=True)
        self.worker.start()

    def stop(self) -> None:
        logger.info("traffic_generator_stopping", node_id=self.settings.node_id)
        self.metrics.set_service_up(False)
        self.stop_event.set()
        self._publish_status("offline")
        if self.worker is not None:
            self.worker.join(timeout=5)
        self.client.loop_stop()
        self.client.disconnect()

    def snapshot(self) -> dict[str, Any]:
        metrics = self.metrics.snapshot()
        return {
            "status": "ok" if metrics["service_up"] else "starting",
            "node_id": self.settings.node_id,
            "scenario": self.settings.replay_scenario,
            "mqtt_connected": metrics["mqtt_connected"],
            "rows_loaded": self.rows_loaded,
            "published_flows": metrics["published_flows"],
            "replay_rate": self.settings.replay_rate,
        }

    def _load_feature_names(self) -> list[str]:
        path = self.settings.artifacts_dir / "feature_names.pkl"
        try:
            feature_names = list(joblib.load(path))
        except Exception as exc:  # noqa: BLE001
            fail_startup(f"Cannot load feature_names.pkl from {path}: {exc}")

        if len(feature_names) != 28:
            fail_startup(f"feature_names.pkl must contain 28 features, found {len(feature_names)}")
        logger.info("feature_names_loaded", path=str(path), count=len(feature_names))
        return feature_names

    def _load_dataset(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            fail_startup(f"Replay scenario parquet not found: {self.dataset_path}")

        try:
            dataset = pd.read_parquet(self.dataset_path)
        except Exception as exc:  # noqa: BLE001
            fail_startup(f"Cannot load replay dataset {self.dataset_path}: {exc}")

        missing = sorted(set(self.feature_names) - set(dataset.columns))
        if missing:
            fail_startup(f"Replay dataset {self.dataset_path} missing feature '{missing[0]}'")
        if len(dataset) == 0:
            fail_startup(f"Replay dataset is empty: {self.dataset_path}")

        logger.info(
            "replay_dataset_loaded",
            path=str(self.dataset_path),
            rows=len(dataset),
            columns=len(dataset.columns),
            scenario=self.settings.replay_scenario,
        )
        return dataset

    def _build_client(self) -> mqtt.Client:
        client_id = f"traffic-generator-{self.settings.node_id}"
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        except (AttributeError, TypeError):
            client = mqtt.Client(client_id=client_id)

        if self.settings.mqtt_username:
            client.username_pw_set(self.settings.mqtt_username, self.settings.mqtt_password)

        client.reconnect_delay_set(min_delay=1, max_delay=30)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        return client

    def _connect_with_retry(self) -> None:
        delay = 1.0
        while not self.stop_event.is_set():
            try:
                self.client.connect(self.settings.mqtt_broker, self.settings.mqtt_port, keepalive=30)
                self.client.loop_start()
                return
            except Exception as exc:  # noqa: BLE001
                self.metrics.mark_error(str(exc))
                logger.warning(
                    "mqtt_connect_retry",
                    broker=self.settings.mqtt_broker,
                    port=self.settings.mqtt_port,
                    delay=delay,
                    error=str(exc),
                )
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

    def _on_connect(self, _client: mqtt.Client, _userdata: Any, _flags: Any, reason_code: Any, *_args: Any) -> None:
        if self._reason_code_value(reason_code) != 0:
            self.connected_event.clear()
            self.metrics.set_mqtt_connected(False)
            self.metrics.mark_error(f"mqtt connect failed: {reason_code}")
            logger.error("mqtt_connect_failed", reason_code=str(reason_code))
            return

        self.connected_event.set()
        self.metrics.set_mqtt_connected(True)
        logger.info("mqtt_connected", broker=self.settings.mqtt_broker, port=self.settings.mqtt_port)

    def _on_disconnect(self, _client: mqtt.Client, _userdata: Any, *args: Any) -> None:
        self.connected_event.clear()
        self.metrics.set_mqtt_connected(False)
        reason_code = args[1] if len(args) >= 2 else args[0] if args else "unknown"
        logger.warning("mqtt_disconnected", reason_code=str(reason_code))

    def _replay_loop(self) -> None:
        delay = 1.0 / self.settings.replay_rate
        row_index = 0

        while not self.stop_event.is_set():
            if not self.connected_event.is_set():
                time.sleep(min(delay, 1.0))
                continue

            row = self.dataset.iloc[row_index]
            row_index = (row_index + 1) % self.rows_loaded

            try:
                payload = self._build_flow_message(row)
                publish_result = self.client.publish(self.flow_topic, json.dumps(payload, separators=(",", ":")), qos=1)
                if publish_result.rc != mqtt.MQTT_ERR_SUCCESS:
                    raise RuntimeError(f"publish failed with rc={publish_result.rc}")
                self.metrics.mark_published()
                logger.debug("flow_published", flow_id=payload["flow_id"], topic=self.flow_topic)
            except ValueError as exc:
                self.metrics.mark_skipped("invalid_feature_value", str(exc))
                logger.error("row_skipped", reason="invalid_feature_value", error=str(exc))
            except Exception as exc:  # noqa: BLE001
                self.metrics.mark_skipped("publish_error", str(exc))
                logger.warning("publish_failed", topic=self.flow_topic, error=str(exc))

            time.sleep(delay)

    def _build_flow_message(self, row: pd.Series) -> dict[str, Any]:
        self.sequence += 1
        features = {name: self._feature_value(row, name) for name in self.feature_names}
        flow_id = f"{self.settings.node_id}_{self.settings.replay_scenario}_{self.sequence:06d}"

        return {
            "schema_version": "1.0",
            "event_type": "iot_flow",
            "flow_id": flow_id,
            "node_id": self.settings.node_id,
            "scenario": self.settings.replay_scenario,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "features": features,
            "ground_truth_label_id": int(row["label_id"]) if "label_id" in row.index else None,
        }

    @staticmethod
    def _feature_value(row: pd.Series, name: str) -> float:
        if name not in row.index:
            raise ValueError(f"row missing feature '{name}'")

        value = row[name]
        if pd.isna(value):
            raise ValueError(f"feature '{name}' is NaN")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"feature '{name}' is not finite")
        return numeric_value

    def _publish_status(self, status: str) -> None:
        if not self.connected_event.is_set() and status == "online":
            return
        payload = {
            "schema_version": "1.0",
            "event_type": "traffic_generator_status",
            "node_id": self.settings.node_id,
            "scenario": self.settings.replay_scenario,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "status": status,
        }
        self.client.publish(self.status_topic, json.dumps(payload, separators=(",", ":")), qos=1)

    @staticmethod
    def _reason_code_value(reason_code: Any) -> int:
        value = getattr(reason_code, "value", reason_code)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0 if str(value).lower() == "success" else 1


def configure_logging(settings: Settings) -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level.upper(),
        serialize=settings.log_format.lower() == "json",
        backtrace=False,
        diagnose=False,
    )


def fail_startup(message: str) -> None:
    logger.critical("Cannot start traffic-generator: {}", message)
    sys.exit(1)


settings = Settings.from_env()
configure_logging(settings)
metrics = TrafficGeneratorMetrics()
replay_service = TrafficReplayService(settings, metrics)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> Any:
    replay_service.start()
    try:
        yield
    finally:
        replay_service.stop()


app = FastAPI(title="QI-FL-IDS-IoT traffic-generator", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return replay_service.snapshot()


@app.get("/metrics")
def prometheus_metrics() -> Response:
    return Response(
        content=metrics.prometheus_text(settings.node_id, settings.replay_scenario),
        media_type="text/plain",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
