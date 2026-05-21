# P15 Online Replay and Operational Deployment Validation Audit

Date: 2026-05-21

## Scope

Audited files and services:

- `experiments/qi-fl-ids-iot-final/deployment/docker-compose.final.yml`
- `experiments/qi-fl-ids-iot-final/deployment/final_ids_api/`
- `experiments/qi-fl-ids-iot-final/dashboard/`
- `services/traffic-generator/`
- `services/iot-node/`
- `services/mosquitto/`
- `services/monitoring/`
- `services/dashboard/`
- `services/edge-ids-gateway/`

## Operational Findings

### 1. How the traffic-generator publishes flows

`services/traffic-generator/replay.py` loads a parquet replay scenario and publishes JSON flow messages to MQTT.

- Flow topic: `ids/flows/{target_node_id}`
- Status topic: `ids/status/{NODE_ID}`
- Default/final compose scenario: `deployment_15`
- Final compose data volume: `experiments/fl-iot-ids-v3/outputs/model_factory_30rounds/deployment_data`
- Final compose artifacts volume: `experiments/fl-iot-ids-v3/outputs/deployment/baseline_fedavg_normal_classweights`
- Published payload includes:
  - `flow_id`
  - `node_id`
  - `timestamp`
  - `features`
  - optional `ground_truth_label_id`

### 2. Existing MQTT topics

Observed topic contracts:

- `ids/flows/{node_id}`: traffic-generator flow input for inference nodes.
- `ids/predictions/{node_id}`: prediction output from `services/iot-node` or `services/edge-ids-gateway`.
- `ids/alerts/{node_id}`: alert output from `services/iot-node` or `services/edge-ids-gateway`.
- `ids/status/{node_id}`: status output from traffic-generator and iot-node.
- `ids/status/gateway/{node_id}`: status output from edge gateway.
- `iot/raw/node1`: raw-event input for the optional edge gateway.
- `iot/accepted/node1` and `iot/blocked/node1`: optional edge gateway decisions.

### 3. Existing consumers for `ids/flows/#`

`services/iot-node/collector.py` consumes `ids/flows/{NODE_ID}` and publishes predictions/alerts. However, this service is not part of the isolated P14 final compose.

`services/edge-ids-gateway/` can publish predictions and alerts too, but it consumes raw events on `iot/raw/node1`, not `ids/flows/#`, and is not part of the P14 final compose.

### 4. final-ids-api path

`experiments/qi-fl-ids-iot-final/deployment/final_ids_api/app.py` exposes the final P8 FedAvg + QGA model through HTTP:

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /model/info`
- `POST /predict`
- `POST /predict/batch`

The API accepts either:

- 12 selected scaled QGA features, or
- 28 original scaled L1 features, then applies the deployment feature mask internally.

It does not currently subscribe to MQTT and does not publish `ids/predictions/#` or `ids/alerts/#`.

### 5. Predictions and alerts in the final stack

The P14 final stack currently starts:

- `final-ids-api`
- `dashboard-p13`
- `mosquitto`
- `prometheus`
- `grafana`
- optional `traffic-generator`

This stack does not include a final MQTT inference bridge. Therefore, traffic-generator can publish `ids/flows/#`, but predictions/alerts require either the historical `iot-node`, the optional edge gateway, or a future bridge from MQTT flows to `final-ids-api`.

### 6. Prometheus scraping

`experiments/qi-fl-ids-iot-final/deployment/monitoring/prometheus.final.yml` scrapes:

- `final-ids-api:8014`
- `traffic-generator:8000`

The new P15 online-validator exposes `/metrics`; it can be queried directly and can be added to Prometheus in a later monitoring hardening step if desired.

### 7. Missing pieces for complete online replay

Current gaps:

- No final MQTT bridge from `ids/flows/#` to `final-ids-api /predict`.
- No final service publishing P8/P8+QGA binary predictions to `ids/predictions/#`.
- No final alert publisher for the P8 deployment model.
- No online evidence collector for endpoint readiness and measured replay metrics.
- No standardized online replay CSV/JSON outputs.

## Recommendation

Implement P15 in two layers:

1. HTTP replay validator: replay scaled L1 test/deployment rows directly into `final-ids-api /predict`, measure latency and online IDS metrics, and write reproducible reports.
2. MQTT observer: add a lightweight online-validator service that subscribes to `ids/flows/#`, `ids/predictions/#`, `ids/alerts/#`, and `ids/status/#`, exposes Prometheus metrics, and documents whether the MQTT path is producing inference outputs.

This avoids modifying the final model, avoids retraining, preserves the P8/P14 deployment decision, and makes the remaining MQTT bridge explicit future work rather than silently mixing old inference nodes with the final P8 model.
