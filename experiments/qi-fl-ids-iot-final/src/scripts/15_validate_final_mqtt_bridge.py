from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any


FINAL_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = FINAL_DIR.parents[1]
DEFAULT_REPORTS_DIR = FINAL_DIR / "outputs" / "reports"
DEFAULT_FEATURE_SCHEMA = FINAL_DIR / "deployment" / "l1_final" / "feature_schema.json"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def http_get_json(url: str, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return {
                "ok": 200 <= response.status < 300,
                "status_code": response.status,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "body": json.loads(body) if body else {},
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        parsed: Any
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        return {
            "ok": False,
            "status_code": exc.code,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "body": parsed,
            "error": str(exc),
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "ok": False,
            "status_code": None,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "body": None,
            "error": str(exc),
        }


def load_env_password() -> str | None:
    if os.getenv("MQTT_PASSWORD"):
        return os.getenv("MQTT_PASSWORD")
    env_path = REPO_ROOT / "services" / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("MQTT_PASSWORD="):
            return line.split("=", 1)[1].strip().strip("\"'")
    return None


def build_safe_flow_payload(feature_schema_path: Path, node_id: str) -> dict[str, Any]:
    schema = read_json(feature_schema_path)
    selected_features = [str(item) for item in schema.get("selected_features", [])]
    if not selected_features:
        selected_features = [f"feature_{index}" for index in range(12)]
    return {
        "schema_version": "1.0",
        "event_type": "safe_replay_flow",
        "node_id": node_id,
        "flow_id": f"safe_replay_{int(time.time() * 1000)}",
        "timestamp": utc_now(),
        "features": {name: 0.0 for name in selected_features},
        "source": "15_validate_final_mqtt_bridge.py",
        "note": "Synthetic scaled-zero payload for safe bridge validation; not offensive traffic.",
    }


def mqtt_roundtrip(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import paho.mqtt.client as mqtt  # type: ignore
    except ImportError as exc:
        return {"ok": False, "error": f"paho-mqtt is not installed: {exc}"}

    payload = build_safe_flow_payload(args.feature_schema, args.node_id)
    prediction_topic = f"ids/predictions/{args.node_id}"
    alert_topic = f"ids/alerts/{args.node_id}"
    flow_topic = f"ids/flows/{args.node_id}"
    received: list[dict[str, Any]] = []
    got_prediction = Event()

    try:
        try:
            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"p15-bridge-validator-{int(time.time())}",
            )
        except (AttributeError, TypeError):
            client = mqtt.Client(client_id=f"p15-bridge-validator-{int(time.time())}")

        if args.mqtt_username:
            client.username_pw_set(args.mqtt_username, args.mqtt_password)

        def on_connect(client_obj: Any, _userdata: Any, _flags: Any, reason_code: Any, *_extra: Any) -> None:
            value = getattr(reason_code, "value", reason_code)
            if str(value).lower() in {"0", "success"}:
                client_obj.subscribe([(prediction_topic, 1), (alert_topic, 1)])

        def on_message(_client_obj: Any, _userdata: Any, message: Any) -> None:
            try:
                body = json.loads(message.payload.decode("utf-8"))
            except json.JSONDecodeError:
                body = message.payload.decode("utf-8", errors="replace")
            received.append({"topic": str(message.topic), "payload": body})
            if str(message.topic) == prediction_topic:
                got_prediction.set()

        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(args.broker, args.port, keepalive=30)
        client.loop_start()
        time.sleep(0.5)
        result = client.publish(flow_topic, json.dumps(payload), qos=1)
        publish_rc = int(result.rc)
        got_prediction.wait(args.timeout_sec)
        client.loop_stop()
        client.disconnect()

        return {
            "ok": got_prediction.is_set(),
            "published_topic": flow_topic,
            "prediction_topic": prediction_topic,
            "alert_topic": alert_topic,
            "publish_rc": publish_rc,
            "messages_received": received,
            "safe_payload_flow_id": payload["flow_id"],
        }
    except Exception as exc:  # noqa: BLE001 - evidence script should report service state, not crash.
        return {
            "ok": False,
            "published_topic": flow_topic,
            "prediction_topic": prediction_topic,
            "alert_topic": alert_topic,
            "error": str(exc),
            "safe_payload_flow_id": payload["flow_id"],
        }


def write_reports(evidence: dict[str, Any], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "p15_final_mqtt_bridge_validation.json"
    md_path = reports_dir / "p15_final_mqtt_bridge_validation.md"
    json_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    api_ready = evidence["checks"]["final_ids_api_ready"].get("ok")
    bridge_ready = evidence["checks"]["final_mqtt_bridge_ready"].get("ok")
    mqtt_result = evidence.get("mqtt_roundtrip", {})
    lines = [
        "# P15 Final MQTT Bridge Validation",
        "",
        f"Generated at: `{evidence['generated_at']}`",
        "",
        "## Endpoint Checks",
        "",
        f"- Final IDS API `/health`: `{evidence['checks']['final_ids_api_health'].get('ok')}`",
        f"- Final IDS API `/ready`: `{api_ready}`",
        f"- Final MQTT bridge `/health`: `{evidence['checks']['final_mqtt_bridge_health'].get('ok')}`",
        f"- Final MQTT bridge `/ready`: `{bridge_ready}`",
        "",
        "## Safe MQTT Roundtrip",
        "",
        f"- Publish sample enabled: `{evidence['publish_sample']}`",
        f"- Roundtrip prediction observed: `{mqtt_result.get('ok')}`",
        f"- Published topic: `{mqtt_result.get('published_topic')}`",
        f"- Prediction topic: `{mqtt_result.get('prediction_topic')}`",
        f"- Alert topic: `{mqtt_result.get('alert_topic')}`",
        f"- Messages received: `{len(mqtt_result.get('messages_received', []))}`",
        "",
        "This validation uses a synthetic scaled-zero replay payload only. It does not run offensive traffic, retrain a model, or modify scientific results.",
        "",
    ]
    if evidence.get("warnings"):
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in evidence["warnings"])
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the final MQTT bridge with safe endpoint and optional replay checks.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8014")
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8016")
    parser.add_argument("--broker", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--mqtt-username", default=os.getenv("MQTT_USERNAME", "ids_user"))
    parser.add_argument("--mqtt-password", default=load_env_password())
    parser.add_argument("--node-id", default="node1")
    parser.add_argument("--feature-schema", type=Path, default=DEFAULT_FEATURE_SCHEMA)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--publish-sample", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when readiness or roundtrip checks fail.")
    args = parser.parse_args()

    evidence: dict[str, Any] = {
        "generated_at": utc_now(),
        "api_url": args.api_url,
        "bridge_url": args.bridge_url,
        "broker": args.broker,
        "port": args.port,
        "publish_sample": bool(args.publish_sample),
        "checks": {
            "final_ids_api_health": http_get_json(f"{args.api_url.rstrip('/')}/health", args.timeout_sec),
            "final_ids_api_ready": http_get_json(f"{args.api_url.rstrip('/')}/ready", args.timeout_sec),
            "final_mqtt_bridge_health": http_get_json(f"{args.bridge_url.rstrip('/')}/health", args.timeout_sec),
            "final_mqtt_bridge_ready": http_get_json(f"{args.bridge_url.rstrip('/')}/ready", args.timeout_sec),
        },
        "warnings": [],
    }

    if args.publish_sample:
        evidence["mqtt_roundtrip"] = mqtt_roundtrip(args)
    else:
        evidence["mqtt_roundtrip"] = {
            "ok": None,
            "note": "Not executed. Pass --publish-sample to publish one safe synthetic flow.",
        }

    if not args.feature_schema.exists():
        evidence["warnings"].append(f"Feature schema not found: {args.feature_schema}")
    if not evidence["checks"]["final_ids_api_ready"].get("ok"):
        evidence["warnings"].append("Final IDS API is not ready or not reachable.")
    if not evidence["checks"]["final_mqtt_bridge_ready"].get("ok"):
        evidence["warnings"].append("Final MQTT bridge is not ready or not reachable.")
    if args.publish_sample and not evidence["mqtt_roundtrip"].get("ok"):
        evidence["warnings"].append("Safe MQTT roundtrip did not observe a prediction message.")

    write_reports(evidence, args.reports_dir)

    if args.strict:
        endpoints_ok = all(check.get("ok") for check in evidence["checks"].values())
        roundtrip_ok = (not args.publish_sample) or bool(evidence["mqtt_roundtrip"].get("ok"))
        return 0 if endpoints_ok and roundtrip_ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

