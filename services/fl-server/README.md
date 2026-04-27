# fl-server

> Status: P5 mock Flower training profile.

## Role

`fl-server` starts a lightweight Flower FedAvg server for the Docker Compose
`training` profile.

This service validates orchestration only:

- dedicated Docker profile
- server/client startup order
- client fit/evaluate loops
- MLflow tracking availability
- clean separation from Mode A inference

It does not import the validated Multi-tier FL implementation from
`experiments/fl-iot-ids-v3/`.

## Environment

- `FL_SERVER_HOST`: bind host, default `0.0.0.0`
- `FL_SERVER_PORT`: Flower gRPC port, default `8080`
- `FL_NUM_ROUNDS`: number of mock rounds, default `10`
- `TRAINING_MODE`: must be `mock`
- `MLFLOW_TRACKING_URI`: MLflow endpoint, default from compose `http://mlflow:5000`
- `KEEP_SERVER_ALIVE`: keep container running after training, default `true`

## Usage

```bash
cd services
docker compose --profile training up -d --build
docker logs fl-server
```

## Important Scope Note

P5 mock training validates Docker/profile orchestration, not scientific FL
metrics. The real validated Multi-tier FL pipeline remains in
`experiments/fl-iot-ids-v3/`.
