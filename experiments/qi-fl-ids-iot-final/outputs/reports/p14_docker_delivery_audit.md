# P14 Docker Stack and Final Delivery Audit

Branch: `final/quantum-inspired-fl-iot-ids-final`

## Files Inspected

- `services/`
- `services/docker-compose.yml`
- `services/iot-node/`
- `services/traffic-generator/`
- `services/dashboard/`
- `services/monitoring/`
- `services/fl-server/`
- `services/fl-client/`
- `experiments/qi-fl-ids-iot-final/dashboard/`
- `experiments/qi-fl-ids-iot-final/outputs/reports/`
- `experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/final_selected_mask/`

## Existing Services

The repository already contains a broad microservice stack:
- `mosquitto`
- `iot-node`
- `traffic-generator`
- `feature-extractor`
- `edge-ids-gateway`
- `fl-server`
- `fl-client`
- `dashboard`
- `prometheus`
- `grafana`
- `mlflow`
- `node-red`
- `qga-service`

The existing `services/docker-compose.yml` is not modified by P14. It still points to older experiment deployment artifacts in several places and remains useful as the historical service stack.

## Reusable Services

Reusable without modification:
- Mosquitto image/config from `services/mosquitto`
- Traffic generator service from `services/traffic-generator`
- Prometheus/Grafana configs from `services/monitoring`
- P13 dashboard from `experiments/qi-fl-ids-iot-final/dashboard`

Reusable only for future integration:
- `iot-node` and `edge-ids-gateway` have MQTT and inference endpoints, but they are not the simplest final L1 API path for P14.
- `fl-server` and `fl-client` are not used for final inference delivery because P14 must not launch FL training.

## Endpoints Found

Existing services expose:
- dashboard: `/`, `/tab/{tab_name}`, `/api/*`, `/health`
- traffic-generator: `/health`, `/ready`, `/metrics`
- iot-node: `/health`, `/ready`, `/metrics`
- edge-ids-gateway: `/health`, `/ready`, `/diagnostics`, `/metrics`, `/validate/raw`, `/map/features`, `/infer/raw`
- fl-server: `/health`, `/nodes`, `/models`, `/schedule`, `/training/trigger`, `/metrics`

## MQTT Topics Found

Traffic and inference topics include:
- `ids/flows/{node_id}`
- `ids/predictions/{node_id}`
- `ids/alerts/{node_id}`
- `ids/status/{node_id}`
- `iot/raw/node1`
- `iot/accepted/node1`
- `iot/blocked/node1`
- `ids/predictions/node1`
- `ids/alerts/node1`
- `ids/status/gateway/node1`

## Dashboard Choice

Use the P13 dashboard under:

`experiments/qi-fl-ids-iot-final/dashboard/`

Reason:
- it already exposes final P12/P13 evidence;
- it can run locally without Docker;
- it is safer to containerize without changing the existing service dashboard.

## Model Artifacts Available

The final production model is P8 FedAvg + QGA:
- run: `run_20260508_155659`
- architecture: `12 -> 128 -> 64 -> 2`
- mask: `conservative_seed_42`
- checkpoint: `outputs/qga_fedavg_flower_l1/alpha_0.5/k3/runs/run_20260508_155659/checkpoints/last_global_model.pth`
- checkpoint size: about 46 KB

This checkpoint is small enough to package intentionally into the delivery bundle.

Available references:
- QGA final mask JSON files
- feature names JSON
- P13 local evaluation metrics
- P12 ablation metrics

## Missing or Intentionally Not Packaged

Not packaged:
- `test_scaled.npz`
- train/validation/test parquet files
- preprocessed datasets
- partitions
- Flower run logs
- full run directories
- large historical checkpoints outside `deployment/l1_final/artifacts`

The robust scaler pickle is present locally, but the P14 API expects scaled features and documents the scaler source instead of committing the pickle as a deployment artifact.

## Recommended Final Stack

Create an isolated P14 compose file under `deployment/`:
- `mosquitto`
- `final-ids-api`
- `traffic-generator` as optional demo profile
- `dashboard-p13`
- `prometheus`
- `grafana`

Ports:
- final IDS API: `8014`
- dashboard P13: `8013`
- Mosquitto: `1883`
- Prometheus: `9090`
- Grafana: `3000`

## Files That Must Not Be Committed

- datasets
- `outputs/preprocessed`
- `outputs/partitions`
- Flower run directories
- logs
- checkpoints outside the intentionally packaged final deployment artifact
- `.claude/settings.local.json`
