# fl-client

> Status: P5 mock Flower training profile.

## Role

`fl-client` is a lightweight Flower `NumPyClient` used by the Docker Compose
`training` profile. Compose starts three instances:

- `fl-client-1`
- `fl-client-2`
- `fl-client-3`

Each client performs deterministic mock `fit` and `evaluate` steps so the P5
profile can validate orchestration without importing the real FL research code.

## Environment

- `CLIENT_ID`: logical client name, for example `client1`
- `FL_SERVER_ADDRESS`: Flower server address, default `fl-server:8080`
- `TRAINING_MODE`: must be `mock`
- `CLIENT_CONNECT_RETRIES`: retry attempts before failing, default `20`
- `CLIENT_RETRY_DELAY_SECONDS`: base retry delay, default `2.0`
- `MOCK_NUM_EXAMPLES`: reported mock examples per client, default `128`

## Usage

```bash
cd services
docker compose --profile training up -d --build
docker logs fl-client-1
docker logs fl-client-2
docker logs fl-client-3
```

## Important Scope Note

P5 mock clients validate Docker/profile orchestration, not scientific FL metrics.
The real validated Multi-tier FL pipeline remains in
`experiments/fl-iot-ids-v3/`.
