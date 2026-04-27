# traffic-generator

Continuous MQTT replay service for P3. It reads demo parquet subsets, builds
strict 28-feature flow messages from the US1 bundle feature order, and publishes
them to `ids/flows/{NODE_ID}` at a controlled replay rate.

## Runtime

- Health API: `GET /health` on container port `8000`
- Metrics API: `GET /metrics` on container port `8000`
- Publish: `ids/flows/{NODE_ID}`
- Status: `ids/status/{NODE_ID}`

## Environment

| Variable | Default | Description |
|---|---|---|
| `NODE_ID` | `node1` | Target node id and MQTT topic suffix |
| `MQTT_BROKER` | `mosquitto` | MQTT broker host |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_USERNAME` | `ids_user` | MQTT username |
| `MQTT_PASSWORD` | unset | MQTT password from `services/.env` |
| `REPLAY_SCENARIO` | `mixed_chaos` | Parquet filename without `.parquet` |
| `REPLAY_RATE` | `5` | Flow messages per second |
| `DATASET_DIR` | `/data/demo` | Directory containing demo parquet subsets |
| `ARTIFACTS_DIR` | `/artifacts` | Directory containing `feature_names.pkl` |
| `LOG_LEVEL` | `INFO` | Log level |
| `LOG_FORMAT` | `json` | Use JSON logs when set to `json` |

## Scenarios

Official P3 scenarios:

- `normal_traffic`
- `ddos_burst`
- `dos_slow`
- `mirai_wave`
- `recon_scan`
- `mixed_chaos`

Any scenario is accepted if `/data/demo/{REPLAY_SCENARIO}.parquet` exists and
contains the 28 features from `feature_names.pkl`.

## Feature Policy

Each published flow contains exactly the feature names loaded from
`/artifacts/feature_names.pkl`. Extra parquet columns are ignored except
`label_id`, which becomes `ground_truth_label_id`.

Rows with missing, NaN, infinite, or non-numeric feature values are skipped and
counted in `traffic_generator_rows_skipped_total`. No `0.0` imputation is used.

## Manual P3 Checks

From `services/`:

```bash
docker compose up -d mosquitto iot-node-1 traffic-generator
docker compose ps
```

Logs:

```bash
docker logs traffic-generator --tail 80
```

Health:

```bash
curl http://localhost:8010/health
```

Metrics:

```bash
curl http://localhost:8010/metrics
```

Observe MQTT:

```bash
docker exec -it mosquitto mosquitto_sub \
  -h localhost -p 1883 \
  -u ids_user -P "$MQTT_PASSWORD" \
  -t "ids/#"
```

Expected behavior:

- `/health` returns `status=ok`, `mqtt_connected=true`, and increasing `published_flows`
- MQTT flow messages are published to `ids/flows/node1`
- `iot-node-1` publishes predictions and alerts according to the selected scenario
- `/metrics` exposes `traffic_generator_*` series
