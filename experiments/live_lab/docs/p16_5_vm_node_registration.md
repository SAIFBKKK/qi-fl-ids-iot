# P16.5 VM Node Registration

P16.5 uses the existing live-lab-controller endpoint:

`POST /register-node`

No new endpoint is introduced.

## Server

- `SERVER_IP=192.168.56.1`
- Controller URL: `http://192.168.56.1:8020`

## VM1: iot-drone-sitl

Expected registration payload:

```json
{
  "node_id": "iot-drone-sitl",
  "cpu_count": 1,
  "ram_gb": 1.0,
  "device_type": "drone_sitl",
  "mqtt_topic": "ids/flows/iot-drone-sitl"
}
```

Expected tier: `weak`.

Command from VM1 after pulling the branch:

```bash
cd ~/qi-fl-ids-iot
git pull
python3 experiments/live_lab/nodes/iot_rpi_weak/run_node.py --server-url http://192.168.56.1:8020 --register
```

## VM2: iot-smart-watch-medium

Expected registration payload:

```json
{
  "node_id": "iot-smart-watch-medium",
  "cpu_count": 1,
  "ram_gb": 1.5,
  "device_type": "smart_watch_like",
  "mqtt_topic": "ids/flows/iot-smart-watch-medium"
}
```

Expected tier: `medium`.

The tier is logical for the demonstration, not only based on observed VM setup
resources. `smart_watch_like` maps deterministically to `medium`.

Command from VM2 after pulling the branch:

```bash
cd ~/qi-fl-ids-iot
git pull
python3 experiments/live_lab/nodes/iot_smart_watch_medium/run_node.py --server-url http://192.168.56.1:8020 --register
```

## Verification

From either VM or the server:

```bash
curl http://192.168.56.1:8020/nodes
curl http://192.168.56.1:8020/assignments
```

Expected:

- `/nodes` count includes both nodes.
- `/assignments` includes `iot-drone-sitl` with `weak`.
- `/assignments` includes `iot-smart-watch-medium` with `medium`.

## Scope

P16.5 performs registration only. It does not start packet capture, Kali
scenarios, replay, training, or Flower.


