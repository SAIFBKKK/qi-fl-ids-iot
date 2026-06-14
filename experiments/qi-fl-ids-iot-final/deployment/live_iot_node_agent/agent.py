from __future__ import annotations

import argparse
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import requests

from flow_replay import INPUT_MODE_LENGTHS, build_flow_payload
from hardware_profiler import collect_hardware_profile
from mqtt_client import LiveLabMQTTClient


class HealthState:
    def __init__(self) -> None:
        self.registered = False
        self.assigned_tier = "unknown"
        self.published_flows = 0


def register_node(server_url: str, node_id: str, device_type: str) -> dict[str, Any]:
    profile = collect_hardware_profile(node_id=node_id, device_type=device_type)
    response = requests.post(
        f"{server_url.rstrip('/')}/register-node",
        json=profile.as_registration_payload(),
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def start_health_server(port: int, state: HealthState) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name.
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            body = (
                "{"
                f'"status":"ok","registered":{str(state.registered).lower()},'
                f'"assigned_tier":"{state.assigned_tier}","published_flows":{state.published_flows}'
                "}"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = HTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def print_alert(topic: str, payload: dict[str, Any]) -> None:
    severity = payload.get("severity", "unknown")
    flow_id = payload.get("flow_id", "unknown")
    probability = payload.get("probability_attack", payload.get("confidence", "unknown"))
    print(f"[alert] topic={topic} severity={severity} flow_id={flow_id} score={probability}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P16 live IoT node agent for controlled MQTT flow replay.")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--mqtt-host", required=True)
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-username", default=os.getenv("MQTT_USERNAME", "ids_user"))
    parser.add_argument("--mqtt-password", default=os.getenv("MQTT_PASSWORD"))
    parser.add_argument("--input-mode", choices=sorted(INPUT_MODE_LENGTHS), default="selected_12_scaled")
    parser.add_argument("--scenario", default="benign")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--device-type", default="iot_vm")
    parser.add_argument("--serve-health", action="store_true")
    parser.add_argument("--health-port", type=int, default=8021)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = HealthState()
    health_server = start_health_server(args.health_port, state) if args.serve_health else None

    assignment = register_node(args.server_url, args.node_id, args.device_type)
    state.registered = True
    state.assigned_tier = str(assignment["assigned_tier"])
    print(f"[register] node={args.node_id} tier={state.assigned_tier}", flush=True)

    publish_topic = str(assignment.get("mqtt_publish_topic") or f"ids/flows/{args.node_id}")
    alert_topic = str(assignment.get("mqtt_alert_topic") or f"ids/alerts/{args.node_id}")
    client = LiveLabMQTTClient(
        host=args.mqtt_host,
        port=args.mqtt_port,
        username=args.mqtt_username,
        password=args.mqtt_password,
        client_id=f"p16-agent-{args.node_id}",
    )
    client.connect(alert_topic=alert_topic, on_alert=print_alert)

    try:
        for sequence in range(1, args.count + 1):
            payload = build_flow_payload(args.node_id, args.input_mode, args.scenario, sequence)
            client.publish_flow(publish_topic, payload)
            state.published_flows += 1
            print(f"[flow] published flow_id={payload['flow_id']} topic={publish_topic}", flush=True)
            time.sleep(max(args.interval_sec, 0.0))
        print(f"[listen] subscribed to {alert_topic}; press Ctrl+C to stop", flush=True)
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[stop] interrupted", flush=True)
    finally:
        client.stop()
        if health_server is not None:
            health_server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

