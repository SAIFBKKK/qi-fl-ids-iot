# P14 Docker Stack and Final Delivery

## Objective

P14 packages the final L1 IDS deployment path for the project:

- production model: P8 FedAvg + QGA
- dashboard: P13 evidence/evaluation dashboard
- API: FastAPI L1 inference service
- optional demo infrastructure: Mosquitto, traffic generator, Prometheus, Grafana

P14 does not retrain models and does not run Flower full.

## Final Model

Primary deployment model:

```text
P8 FedAvg + QGA L1
selected_mask_id = conservative_seed_42
features_count = 12
architecture = 12 -> 128 -> 64 -> 2
labels = normal:0, attack:1
threshold = 0.4
```

Research alternative:

```text
P9 QIFA + QGA L1
```

The alternative is retained in reports/dashboard but is not the main production artifact.

## Services

`final-ids-api`
- FastAPI inference service
- port `8014`
- endpoints: `/health`, `/ready`, `/metrics`, `/model/info`, `/predict`, `/predict/batch`

`dashboard-p13`
- P13 dashboard
- port `8013`

`mosquitto`
- MQTT broker
- port `1883`

`traffic-generator`
- optional profile `demo`
- port `8010`

`prometheus`
- port `9090`

`grafana`
- port `3000`

## Local Commands

Build the deployment bundle:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/14_build_delivery_manifest.py
```

Verify delivery setup:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/14_verify_delivery_setup.py
```

Smoke-test the API without Docker:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/14_smoke_test_final_api.py
```

Run API locally:

```powershell
cd experiments/qi-fl-ids-iot-final/deployment/final_ids_api
python app.py
```

Run dashboard locally:

```powershell
cd experiments/qi-fl-ids-iot-final/dashboard
python app.py
```

## Docker Commands

Run final stack:

```powershell
cd experiments/qi-fl-ids-iot-final/deployment
docker compose -f docker-compose.final.yml up --build
```

Run with optional traffic generator:

```powershell
docker compose -f docker-compose.final.yml --profile demo up --build
```

## Example Prediction

The API accepts 12 selected scaled features or 28 original scaled features:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8014/predict `
  -ContentType "application/json" `
  -Body '{"features":[0,0,0,0,0,0,0,0,0,0,0,0]}'
```

## Readiness

- `/health` stays `ok` even when the model artifact is unavailable.
- `/ready` is true only when the packaged model can be loaded.
- `/model/info` exposes selected model metadata and bundle manifest.

## Limits

- No datasets are included in the Docker bundle.
- No global test holdout is shipped.
- Raw packet parsing is not part of the final API; it expects scaled CIC-IoT feature vectors.
- P11 FedTN/MPS remains a dry-run/structural compression study.
- The existing `services/docker-compose.yml` is not modified.

## Relation With P13

P13 remains the evidence dashboard. P14 adds a containerizable inference API and final compose file while keeping P13 launchable locally and in Docker.
