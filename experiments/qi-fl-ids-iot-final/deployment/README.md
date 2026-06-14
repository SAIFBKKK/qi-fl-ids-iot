# P14 Final Delivery

This folder contains the final Docker-oriented delivery for the Quantum-Inspired FL IDS L1 model.

## Contents

- `l1_final/`: production model bundle for P8 FedAvg + QGA
- `final_ids_api/`: FastAPI inference service
- `docker-compose.final.yml`: isolated final stack
- `monitoring/prometheus.final.yml`: P14 Prometheus scrape config

## Build and Verify

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/14_build_delivery_manifest.py
python experiments/qi-fl-ids-iot-final/src/scripts/14_verify_delivery_setup.py
python experiments/qi-fl-ids-iot-final/src/scripts/14_smoke_test_final_api.py
```

## Local API

```powershell
cd experiments/qi-fl-ids-iot-final/deployment/final_ids_api
python app.py
```

Open:

```text
http://127.0.0.1:8014/health
http://127.0.0.1:8014/ready
http://127.0.0.1:8014/model/info
```

## Docker

```powershell
cd experiments/qi-fl-ids-iot-final/deployment
docker compose -f docker-compose.final.yml up --build
```

Optional demo traffic profile:

```powershell
docker compose -f docker-compose.final.yml --profile demo up --build
```

P14 does not include datasets, test holdouts, FL runs, or training artifacts beyond the intentionally packaged final L1 model.
