# P16.2 Server Docker Desktop and Local Network Readiness Plan

Date: 2026-05-26
Branch: `final/quantum-inspired-fl-iot-ids-final`

## Goal

Prepare the central server PC so the future VirtualBox VMs can reach the final
IDS lab services. This step validates configuration and documents the network.
It does not create VMs, start capture, run training, run Flower, or modify
P8-P16 results.

## Required Server Services

The future server stack is defined in:

`experiments/qi-fl-ids-iot-final/deployment/docker-compose.final.yml`

Required services:

- `mosquitto`
- `final-ids-api`
- `final-mqtt-bridge`
- `online-validator`
- `live-lab-controller`
- `dashboard-p13`
- `prometheus`
- `grafana`

## Ports to Expose

- MQTT Mosquitto: `1883`
- final IDS API: `8014`
- final MQTT bridge: `8016`
- online validator: `8015`
- live lab controller: `8020`
- dashboard P13: `8013`
- Prometheus: `9090`
- Grafana: `3000`

## Endpoints to Test

When the stack is running:

- `http://127.0.0.1:8014/ready`
- `http://127.0.0.1:8016/ready`
- `http://127.0.0.1:8015/ready`
- `http://127.0.0.1:8020/health`
- `http://127.0.0.1:8020/nodes`
- `http://127.0.0.1:8013/health`

Endpoint failures are warnings in P16.2 if Docker Desktop is not running.

## Server IP for Future VMs

Use the IP of the Windows host on the VirtualBox Host-Only Network. Keep
`SERVER_IP` as a placeholder until the adapter is configured.

Example later check:

```powershell
ipconfig
```

Look for the VirtualBox Host-Only adapter IPv4 address and place it in:

- `experiments/live_lab/configs/server.env.example`
- VM-side node configs
- VM connection card

## Recommended Network Choice

Use a VirtualBox Host-Only Network for the lab:

- local to the server and VMs;
- predictable addressing;
- no external target machines;
- easier Windows Firewall scoping.

## Future VM Location

Create VMs later under:

`E:\VirtualBox VMs\qi-fl-ids-iot-live-lab\`

Planned VM names:

- `iot-rpi-weak`
- `iot-smart-watch-medium`

## Windows Firewall Checklist

Allow inbound TCP from the VirtualBox Host-Only subnet to:

- `1883`
- `8013`
- `8014`
- `8015`
- `8016`
- `8020`
- `9090`
- `3000`

Keep the rule scoped to the host-only/private lab network where possible.

## Recommended Docker Commands

Configuration only:

```powershell
cd experiments\qi-fl-ids-iot-final\deployment
docker compose -f docker-compose.final.yml --profile online --profile live-lab config
```

Later, when Docker Desktop is intentionally running:

```powershell
cd experiments\qi-fl-ids-iot-final\deployment
docker compose -f docker-compose.final.yml --profile online --profile live-lab up -d --build
docker compose -f docker-compose.final.yml --profile online --profile live-lab ps
```

P16.2 does not require starting the stack.

