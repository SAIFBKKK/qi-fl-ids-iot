# P16.2 Server Readiness

- Generated at: `2026-05-26T15:29:38.599535Z`
- Overall status: `OK`
- Compose file exists: `True`
- Required services defined: `True`
- Docker daemon reachable: `True`
- Compose config command: `docker compose -f C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\deployment\docker-compose.final.yml --profile online --profile live-lab config`
- Compose config status: `True`

## Services

- `mosquitto`: `True`
- `final-ids-api`: `True`
- `final-mqtt-bridge`: `True`
- `online-validator`: `True`
- `live-lab-controller`: `True`
- `dashboard-p13`: `True`
- `prometheus`: `True`
- `grafana`: `True`

## Endpoints

- `final_ids_api_ready`: `http://127.0.0.1:8014/ready`, reachable=`True`, status=`200`
- `final_mqtt_bridge_ready`: `http://127.0.0.1:8016/ready`, reachable=`True`, status=`200`
- `online_validator_ready`: `http://127.0.0.1:8015/ready`, reachable=`True`, status=`200`
- `live_lab_controller_health`: `http://127.0.0.1:8020/health`, reachable=`True`, status=`200`
- `live_lab_controller_nodes`: `http://127.0.0.1:8020/nodes`, reachable=`True`, status=`200`
- `dashboard_p13_health`: `http://127.0.0.1:8013/health`, reachable=`True`, status=`200`

## Warnings

- None.

## VM Network

- Use `SERVER_IP` from the Windows VirtualBox Host-Only adapter.
- Future VM root: `E:\VirtualBox VMs\qi-fl-ids-iot-live-lab\`.
