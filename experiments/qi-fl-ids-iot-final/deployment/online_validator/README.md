# P15 Online Validator

Lightweight MQTT observer for P15 online replay evidence.

The service does not train, infer, or modify the final model. It subscribes to
the operational IDS topics and exposes counters through HTTP and Prometheus.

## Topics

Default subscriptions:

- `ids/flows/#`
- `ids/predictions/#`
- `ids/alerts/#`
- `ids/status/#`

## Endpoints

- `GET /health`
- `GET /ready`
- `GET /summary`
- `GET /metrics`

## Docker Compose

From `experiments/qi-fl-ids-iot-final/deployment`:

```powershell
docker compose -f docker-compose.final.yml --profile online up --build online-validator
```

The service is intentionally profile-gated so the stable P14 stack remains
unchanged unless online validation is requested.
