# iot-node

MQTT inference node for P2. The service subscribes to one node-specific flow topic,
preprocesses incoming JSON safely, runs inference, publishes predictions, and emits
alerts when confidence crosses the configured threshold.

## Runtime

- Health API: `GET /health` on container port `8000`
- Metrics API: `GET /metrics` on container port `8000`
- Subscribe: `ids/flows/{NODE_ID}`
- Publish: `ids/predictions/{NODE_ID}`, `ids/alerts/{NODE_ID}`, `ids/status/{NODE_ID}`

## Environment

| Variable | Default | Description |
|---|---|---|
| `NODE_ID` | `node1` | Node identifier used in MQTT topics |
| `MQTT_BROKER` | `mosquitto` | MQTT broker host |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_USERNAME` | `ids_user` | MQTT username |
| `MQTT_PASSWORD` | unset | MQTT password from `services/.env` |
| `INFERENCE_THRESHOLD` | `0.5` | Alert threshold on prediction confidence |
| `MODEL_PATH` | `/artifacts/global_model.pth` | PyTorch state dict path |
| `SCALER_PATH` | `/artifacts/scaler.pkl` | StandardScaler path |
| `LABEL_MAPPING_PATH` | `/artifacts/label_mapping.json` | Label mapping path |
| `LOG_LEVEL` | `INFO` | Log level |
| `LOG_FORMAT` | `json` | Use JSON logs when set to `json` |

## Inference Mode

At startup, the node loads `label_mapping.json`, `feature_names.pkl`, `scaler.pkl`,
`model_config.json`, and `global_model.pth`. If any bundle asset cannot be loaded,
the service logs a critical error and exits immediately.

Flows must include exactly the 28 features from `feature_names.pkl`. Missing or
unexpected features are rejected without publishing a prediction.

## Message Formats

Input flow:

```json
{
  "schema_version": "1.0",
  "flow_id": "flow_001",
  "node_id": "node1",
  "timestamp": "2026-04-27T13:00:00Z",
  "features": {
    "feature_a": 0.1,
    "feature_b": 2.3
  }
}
```

Prediction output:

```json
{
  "schema_version": "1.0",
  "event_type": "ids_prediction",
  "node_id": "node1",
  "timestamp": "2026-04-27T13:00:00Z",
  "flow_id": "flow_001",
  "predicted_label": "DDoS-ICMP_Flood",
  "predicted_label_id": 6,
  "confidence": 0.635455,
  "is_alert": true,
  "model_version": "baseline_fedavg_normal_classweights"
}
```

Alert output:

```json
{
  "schema_version": "1.0",
  "event_type": "ids_alert",
  "node_id": "node1",
  "timestamp": "2026-04-27T13:00:00Z",
  "flow_id": "flow_001",
  "predicted_label": "DDoS-ICMP_Flood",
  "predicted_label_id": 6,
  "confidence": 0.635455,
  "severity": "low",
  "source_topic": "ids/flows/node1",
  "model_version": "baseline_fedavg_normal_classweights"
}
```

Severity mapping:

| Confidence | Severity |
|---:|---|
| `< 0.70` | `low` |
| `0.70 - 0.85` | `medium` |
| `0.85 - 0.95` | `high` |
| `>= 0.95` | `critical` |

## Manual P2 Checks

From `services/`:

```bash
docker compose up -d mosquitto iot-node-1
docker compose ps
curl http://localhost:8001/health
```

Subscribe to all IDS topics:

```bash
docker exec -it mosquitto mosquitto_sub \
  -h localhost -p 1883 \
  -u ids_user -P "$MQTT_PASSWORD" \
  -t "ids/#"
```

Publish test flows from a demo subset:

```bash
python services/scripts/test_publish.py --subset ddos_burst --count 5 --rate 1 --node-id node1
```

Expected behavior:

- `/health` returns `status: ok`
- A prediction is published to `ids/predictions/node1`
- An alert is published to `ids/alerts/node1` when confidence is above threshold and the label is not benign
- `ids/status/node1` is retained as `online` while the node is running
