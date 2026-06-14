# P16 Live Lab Controller

Lightweight server-side registry for the P16 live lab. It assigns each IoT VM to
a hardware tier and returns the MQTT topics expected by the final online path:

`ids/flows/{node_id}` -> `final-mqtt-bridge` -> `final-ids-api` -> `ids/predictions/{node_id}` / `ids/alerts/{node_id}`.

## Endpoints

- `GET /health`
- `GET /ready`
- `POST /register-node`
- `GET /nodes`
- `GET /assignments`
- `GET /metrics`

## Registration Payload

```json
{
  "node_id": "iot-drone-sitl",
  "hostname": "iot-drone-sitl-node",
  "cpu_count": 2,
  "ram_gb": 2,
  "device_type": "drone_sitl",
  "mqtt_topic": "ids/flows/iot-drone-sitl"
}
```

## Assignment Policy

- `weak` if `cpu_count <= 2` or `ram_gb <= 4`
- `medium` if `cpu_count <= 4` or `ram_gb <= 8`
- `powerful` otherwise

The controller advertises both final API input modes:

- `selected_12_scaled`
- `original_28_scaled`


