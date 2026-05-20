# P14 Docker Stack and Final Delivery Plan

## Architecture

P14 delivers an isolated final stack under:

`experiments/qi-fl-ids-iot-final/deployment/`

The historical `services/docker-compose.yml` remains untouched.

Final services:
- `final-ids-api`
- `dashboard-p13`
- `mosquitto`
- `traffic-generator` optional demo profile
- `prometheus`
- `grafana`

## Production Model

Primary model:
- P8 FedAvg + QGA L1
- `selected_mask_id=conservative_seed_42`
- 12 selected features
- architecture `12 -> 128 -> 64 -> 2`
- labels `normal=0`, `attack=1`
- threshold from validation-only P8 artifact: `0.4`

Research alternative:
- P9 QIFA + QGA L1
- report-only in P14 delivery

## Ports

- `final-ids-api`: host `8014`, container `8014`
- `dashboard-p13`: host `8013`, container `8013`
- `mosquitto`: host `1883`, container `1883`
- `prometheus`: host `9090`, container `9090`
- `grafana`: host `3000`, container `3000`
- optional `traffic-generator`: host `8010`, container `8000`

## Volumes

The final API mounts:

`./l1_final:/app/l1_final:ro`

Mosquitto and monitoring reuse existing service configs as read-only volumes.

## Environment

`final-ids-api`:
- `DEPLOYMENT_BUNDLE_DIR=/app/l1_final`

`traffic-generator` demo profile:
- `MQTT_BROKER=mosquitto`
- `MQTT_PORT=1883`
- `MQTT_USERNAME=ids_user`
- `MQTT_PASSWORD` from environment if needed

## API Readiness

Required checks:
- `GET /health` returns `ok`
- `GET /ready` returns true when `model.pth` is packaged
- `GET /model/info` returns selected P8 model metadata
- `POST /predict` works for a 12-feature synthetic input when the model is packaged

## Launch Commands

Build bundle and verify locally:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/14_build_delivery_manifest.py
python experiments/qi-fl-ids-iot-final/src/scripts/14_verify_delivery_setup.py
python experiments/qi-fl-ids-iot-final/src/scripts/14_smoke_test_final_api.py
```

Run local API:

```powershell
cd experiments/qi-fl-ids-iot-final/deployment/final_ids_api
python app.py
```

Run local dashboard:

```powershell
cd experiments/qi-fl-ids-iot-final/dashboard
python app.py
```

Docker stack:

```powershell
cd experiments/qi-fl-ids-iot-final/deployment
docker compose -f docker-compose.final.yml up --build
```

Optional traffic generator profile:

```powershell
docker compose -f docker-compose.final.yml --profile demo up --build
```

## Limits

- No FL training is launched.
- P11 FedTN/MPS remains a structural dry-run in the dashboard.
- The API expects scaled CIC-IoT feature vectors. Raw packet parsing remains a future service integration step.
- The test holdout is not packaged in the Docker delivery.
