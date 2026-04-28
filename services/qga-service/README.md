# qga-service

Optional Quantum-Inspired Optimization API for the `preprocessing` Docker
profile.

P6C intentionally ships a deterministic QGA stub, not a full quantum genetic
algorithm. The goal is to validate the microservice contract, observability, and
extension point without depending on the scientific dataset.

## Run

```bash
cd services
docker compose --profile preprocessing up -d --build qga-service
```

## Endpoints

- `GET /health`
- `POST /optimize`
- `GET /metrics`

Example request:

```bash
curl -X POST http://localhost:8020/optimize \
  -H "Content-Type: application/json" \
  -d '{"available_features":28,"latency_budget_ms":5.0,"energy_budget":0.75,"risk_tolerance":0.4}'
```

Example response:

```json
{
  "status": "ok",
  "selected_features": ["feature_01", "feature_02", "...", "feature_17"],
  "feature_budget": 17,
  "threshold_suggestion": 0.6,
  "optimization_score": 0.8046,
  "qga_iterations": 20,
  "mode": "deterministic_stub"
}
```

## Metrics

- `qga_requests_total`
- `qga_optimization_latency_seconds`
- `qga_last_score`
- `qga_service_status`

## Environment

- `QGA_DEFAULT_ITERATIONS` default `20`
- `LOG_LEVEL` default `INFO`
- `LOG_FORMAT` default `json`
