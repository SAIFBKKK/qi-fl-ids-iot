# Final MQTT Bridge

`final-mqtt-bridge` connects the safe MQTT replay path to the final scientific
deployment model.

Pipeline:

```text
traffic-generator
  -> MQTT ids/flows/{node}
  -> final-mqtt-bridge
  -> final-ids-api /predict
  -> MQTT ids/predictions/{node}
  -> MQTT ids/alerts/{node}
```

The bridge does not train, tune, or modify the model. It forwards scaled replay
features to the final P8 FedAvg + QGA HTTP API and republishes binary
normal/attack predictions and attack alerts.

## Endpoints

- `GET /health`
- `GET /ready`
- `GET /metrics`

## Environment

| Variable | Default |
|---|---|
| `MQTT_HOST` | `mosquitto` |
| `MQTT_PORT` | `1883` |
| `MQTT_USERNAME` | `ids_user` |
| `MQTT_PASSWORD` | unset |
| `MQTT_SUBSCRIBE_TOPIC` | `ids/flows/#` |
| `FINAL_IDS_API_URL` | `http://final-ids-api:8014` |
| `BRIDGE_NODE_ID` | `final-mqtt-bridge` |
| `FEATURE_SCHEMA_PATH` | `/app/l1_final/feature_schema.json` |

## Prometheus Metrics

- `final_mqtt_bridge_ready`
- `final_mqtt_bridge_mqtt_connected`
- `final_mqtt_bridge_flows_received_total`
- `final_mqtt_bridge_predictions_published_total`
- `final_mqtt_bridge_alerts_published_total`
- `final_mqtt_bridge_prediction_errors_total`
- `final_mqtt_bridge_http_latency_seconds`
- `final_mqtt_bridge_last_message_timestamp_seconds`

## Docker

From `experiments/qi-fl-ids-iot-final/deployment`:

```powershell
docker compose -f docker-compose.final.yml --profile online --profile demo up -d --build
```
