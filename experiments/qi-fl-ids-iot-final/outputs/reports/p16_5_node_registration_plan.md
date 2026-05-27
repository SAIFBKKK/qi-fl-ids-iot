# P16.5 Live Lab Node Registration Plan

Date: 2026-05-27
Branch: `final/quantum-inspired-fl-iot-ids-final`

## Controller Audit

The existing `live-lab-controller` exposes:

- `GET /health`
- `GET /ready`
- `POST /register-node`
- `GET /nodes`
- `GET /assignments`
- `GET /metrics`

P16.5 uses the existing endpoint:

`POST /register-node`

No new endpoint is introduced and the request schema remains compatible:

```json
{
  "node_id": "string",
  "hostname": "string",
  "cpu_count": 1,
  "ram_gb": 1.0,
  "device_type": "raspberry_like",
  "mqtt_topic": "ids/flows/example"
}
```

Manual registration was validated before this correction with `count=2` in
`/nodes` and `/assignments` for:

- `iot-rpi-weak`
- `iot-smart-watch-medium`

## Tier Correction

The initial resource-only tier logic assigned `iot-smart-watch-medium` to
`weak` because its declared setup resources are intentionally small. P16.5 fixes
the live-lab demonstration mapping:

- `raspberry_like` -> `weak`
- `smart_watch_like` -> `medium`
- unknown device types fall back to the existing resource rule

The tier is a logical lab role, not only the observed setup RAM/CPU.

## VM1 Payload

```json
{
  "node_id": "iot-rpi-weak",
  "cpu_count": 1,
  "ram_gb": 1.0,
  "device_type": "raspberry_like",
  "mqtt_topic": "ids/flows/iot-rpi-weak"
}
```

Expected assignment: `weak`.

## VM2 Payload

```json
{
  "node_id": "iot-smart-watch-medium",
  "cpu_count": 1,
  "ram_gb": 1.5,
  "device_type": "smart_watch_like",
  "mqtt_topic": "ids/flows/iot-smart-watch-medium"
}
```

Expected assignment: `medium`.

## Validation Commands

VM1:

```bash
cd ~/qi-fl-ids-iot
git pull
python3 experiments/live_lab/nodes/iot_rpi_weak/run_node.py --server-url http://192.168.56.1:8020 --register
```

VM2:

```bash
cd ~/qi-fl-ids-iot
git pull
python3 experiments/live_lab/nodes/iot_smart_watch_medium/run_node.py --server-url http://192.168.56.1:8020 --register
```

Verification:

```bash
curl http://192.168.56.1:8020/nodes
curl http://192.168.56.1:8020/assignments
```

## Limits

P16.5 does not launch packet capture, Kali scenarios, replay, training, Flower,
Docker lifecycle commands, or model changes. It only prepares and validates node
registration and tier assignment.

