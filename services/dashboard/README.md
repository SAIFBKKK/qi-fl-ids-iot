# QI-FL-IDS-IoT Dashboard

FastAPI dashboard service for the QI-FL-IDS-IoT microservices stack.

## Runtime

- Port: `8090`
- Backend registry: `FL_SERVER_URL`, default `http://fl-server:8080`
- Prometheus: `PROMETHEUS_URL`, default `http://prometheus:9090`

## Tabs

- `1. Réseau IoT`: implemented for Sprint 1.
- `2. Federated Learning`: placeholder for Sprint 2.
- `3. QI vs Classique`: placeholder for Sprint 3.

## Endpoints

- `GET /health`
- `GET /`
- `GET /tab/iot`
- `GET /tab/fl`
- `GET /tab/qi`
- `GET /api/nodes`
- `GET /api/models`
- `POST /api/connect`
