# Server Network Readiness

P16.2 prepares the server-side network for the future live lab VMs. It does not
create VMs in this step.

## Server Requirements

- Docker Desktop installed and running when runtime checks are needed.
- Repository branch: `final/quantum-inspired-fl-iot-ids-final`.
- Compose file:
  `experiments/qi-fl-ids-iot-final/deployment/docker-compose.final.yml`.

## Recommended VirtualBox Network

Use a VirtualBox Host-Only Network so the Windows server and the two VMs share a
local isolated lab network.

Future VM storage location:

`E:\VirtualBox VMs\qi-fl-ids-iot-live-lab\`

Planned VMs:

- `iot-rpi-weak`
- `iot-smart-watch-medium`

## Ports to Open in Windows Firewall

Allow inbound TCP from the Host-Only subnet to:

- `1883`: MQTT Mosquitto
- `8013`: dashboard P13
- `8014`: final IDS API
- `8015`: online validator
- `8016`: final MQTT bridge
- `8020`: live lab controller
- `9090`: Prometheus
- `3000`: Grafana

Scope rules to the private/host-only lab network where possible.

## Find the Server IP

On the Windows server:

```powershell
ipconfig
```

Find the IPv4 address for the VirtualBox Host-Only adapter. Use that value as
`SERVER_IP` in VM configs and in the VM connection card.

## Server-Side Checks

Configuration check:

```powershell
cd experiments\qi-fl-ids-iot-final\deployment
docker compose -f docker-compose.final.yml --profile online --profile live-lab config
```

Runtime checks, later when Docker Desktop is intentionally running:

```powershell
curl.exe http://127.0.0.1:8014/ready
curl.exe http://127.0.0.1:8016/ready
curl.exe http://127.0.0.1:8015/ready
curl.exe http://127.0.0.1:8020/health
curl.exe http://127.0.0.1:8020/nodes
curl.exe http://127.0.0.1:8013/health
```

## Future VM Checks

After the VMs are created and attached to the Host-Only network:

```bash
curl http://SERVER_IP:8020/health
curl http://SERVER_IP:8014/ready
curl http://SERVER_IP:8013/health
```

MQTT checks will be added after credentials and broker reachability are
confirmed. No VM creation happens in P16.2.

