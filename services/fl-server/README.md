# fl-server

> Status: P6A-lite training dispatcher.

## Role

`fl-server` dispatches the Docker Compose `training` profile.

- `TRAINING_MODE=mock` starts the lightweight P5 Flower FedAvg server.
- `TRAINING_MODE=real` launches the scientific runner mounted from
  `experiments/fl-iot-ids-v3`.

Mock mode validates orchestration only:

- dedicated Docker profile
- server/client startup order
- client fit/evaluate loops
- MLflow tracking availability
- clean separation from Mode A inference

Real mode does not copy or modify the validated Multi-tier FL implementation.
It executes:

```bash
python -m src.scripts.run_experiment \
  --experiment exp_v4_multitier_fedavg_normal_classweights \
  --rounds ${REAL_FL_ROUNDS}
```

## Environment

- `FL_SERVER_HOST`: bind host, default `0.0.0.0`
- `FL_SERVER_PORT`: Flower gRPC port, default `8080`
- `FL_NUM_ROUNDS`: number of mock rounds, default `10`
- `TRAINING_MODE`: `mock` or `real`, default `mock`
- `REAL_FL_EXPERIMENT`: real runner experiment name
- `REAL_FL_ROUNDS`: real runner round override, default `1`
- `REAL_FL_WORKDIR`: bind-mounted scientific project path
- `MLFLOW_TRACKING_URI`: MLflow endpoint, default from compose `http://mlflow:5000`
- `KEEP_SERVER_ALIVE`: keep container running after training, default `true`

## Usage

```bash
cd services
docker compose --profile training up -d --build
docker logs fl-server
```

Real mode:

```bash
cd services
TRAINING_MODE=real REAL_FL_ROUNDS=1 docker compose --profile training up -d --build
docker logs fl-server
```

## Important Scope Note

P5 mock training validates Docker/profile orchestration, not scientific FL
metrics. P6A-lite real mode validates that the Compose profile can invoke the
real simulation-based Multi-tier runner. It is not yet a multi-container real FL
client deployment.
