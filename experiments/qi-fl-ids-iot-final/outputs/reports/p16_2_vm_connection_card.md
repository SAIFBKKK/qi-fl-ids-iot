# P16.2 VM Connection Card

Set `SERVER_IP` to the Windows host IPv4 address on the VirtualBox Host-Only
Network.

## Server Values

- `SERVER_IP=<SERVER_IP>`
- `MQTT_HOST=<SERVER_IP>`
- `MQTT_PORT=1883`
- `LIVE_LAB_CONTROLLER_URL=http://<SERVER_IP>:8020`
- `FINAL_IDS_API_URL=http://<SERVER_IP>:8014`
- `FINAL_MQTT_BRIDGE_URL=http://<SERVER_IP>:8016`
- `DASHBOARD_URL=http://<SERVER_IP>:8013`
- `GRAFANA_URL=http://<SERVER_IP>:3000`
- `PROMETHEUS_URL=http://<SERVER_IP>:9090`

## Node IDs

- `iot-rpi-weak`
- `iot-smart-watch-medium`

## MQTT Topics

- `ids/flows/iot-rpi-weak`
- `ids/flows/iot-smart-watch-medium`
- `ids/predictions/{node_id}`
- `ids/alerts/{node_id}`
- `ids/status/{node_id}`

## VM Notes

- VM1: `iot-rpi-weak`, raspberry-like weak node, server-side inference.
- VM2: `iot-smart-watch-medium`, smart-watch-like medium node, future edge
  inference.
- Future VM storage root: `E:\VirtualBox VMs\qi-fl-ids-iot-live-lab\`

