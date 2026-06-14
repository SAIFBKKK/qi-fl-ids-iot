# P13 L1 Final Dashboard

This dashboard is an isolated FastAPI app for the final L1 evidence and deployment-readiness view. It does not retrain models and does not modify the service Docker stack.

## Build Assets

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/13_build_dashboard_assets.py
python experiments/qi-fl-ids-iot-final/src/scripts/13_evaluate_dashboard_models.py
```

## Run Locally

The experiment directory contains hyphens, so the safest command is to run from the dashboard folder:

```powershell
cd experiments/qi-fl-ids-iot-final/dashboard
python app.py
```

Then open:

```text
http://127.0.0.1:8013
```

## Modes

- Evidence mode: reads P12, P10, and P11 reports.
- Evaluation mode: evaluates compatible local checkpoints on the L1 test holdout when present.
- Demo/replay mode: reserved for P14 Docker stack integration.

If a model checkpoint is missing or incompatible, the dashboard shows `report_only` and uses validated report metrics.
