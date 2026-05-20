# Final IDS API

FastAPI service for the final L1 deployment model:

- model: P8 FedAvg + QGA
- selected mask: `conservative_seed_42`
- input: 12 selected scaled features, or 28 original scaled features with mask applied by the API

## Local

Build the bundle first:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/14_build_delivery_manifest.py
```

Run:

```powershell
cd experiments/qi-fl-ids-iot-final/deployment/final_ids_api
python app.py
```

Endpoints:
- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /model/info`
- `POST /predict`
- `POST /predict/batch`

If `artifacts/model.pth` is absent from the L1 bundle, `/health` remains `ok` and `/ready` returns `false`.
