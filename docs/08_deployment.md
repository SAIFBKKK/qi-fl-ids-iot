# Deployment

Deployment assets live mainly under:

```text
services/
```

The deployment stack includes:

- Docker Compose services
- MQTT transport
- traffic generation/replay
- feature extraction
- IDS API
- dashboard
- Prometheus
- Grafana

Use `.env.example` files as templates. Never commit real `.env` files.

Example local setup:

```powershell
cd <REPO_ROOT>/services
Copy-Item .env.example .env
```

Then replace placeholders such as `<MQTT_HOST>` and `<SERVER_IP>` locally.

The deployment demo validates integration flow. It does not introduce new model accuracy claims.
