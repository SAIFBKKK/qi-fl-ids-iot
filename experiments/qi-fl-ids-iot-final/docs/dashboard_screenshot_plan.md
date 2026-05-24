# Dashboard Screenshot Plan for Final Report

Date: 2026-05-24

This plan lists the screenshots to capture for the final year project report. It avoids offensive tooling and uses only prepared replay data, health endpoints, Prometheus metrics, Grafana dashboards, and existing P13/P15 validation scripts.

## Pre-Screenshot Setup

### Final Docker stack

From:

```powershell
cd C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\deployment
```

Start the final stack:

```powershell
docker compose -f docker-compose.final.yml up -d --build
```

Optional traffic replay:

```powershell
docker compose -f docker-compose.final.yml --profile demo up -d --build traffic-generator
```

Optional online validator:

```powershell
docker compose -f docker-compose.final.yml --profile online up -d --build online-validator final-mqtt-bridge
```

Full online replay stack with the final MQTT bridge:

```powershell
docker compose -f docker-compose.final.yml --profile online --profile demo up -d --build
```

Safe bridge validation:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/15_validate_final_mqtt_bridge.py --publish-sample
```

### HTTP replay evidence

Use the final P8 FedAvg + QGA model:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/15_run_online_http_replay.py --api-url http://127.0.0.1:8014 --max-rows 1000 --sleep-ms 0 --use-qga-mask
```

Collect endpoint evidence:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/15_collect_online_evidence.py
```

Optionally observe MQTT:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/15_check_mqtt_topics.py --broker 127.0.0.1 --port 1883 --duration 30
```

## Screenshots to Capture

### Screenshot 1 — P13 Final Dashboard Overview

URL:

```text
http://127.0.0.1:8013/
```

Capture:

- Recommended model panel.
- P8 FedAvg + QGA as production L1 model.
- P9 QIFA and P9 QIFA + QGA as alternatives.
- P10 robustness and P11 compression panels if visible.

Caption:

```text
Final L1 deployment dashboard showing the selected P8 FedAvg + QGA model, report-backed alternatives, robustness evidence, and compression evidence.
```

### Screenshot 2 — Final IDS API Health and Model Info

URLs:

```text
http://127.0.0.1:8014/health
http://127.0.0.1:8014/ready
http://127.0.0.1:8014/model/info
```

Capture:

- `ready=true`
- `selected_model=P8 FedAvg + QGA`
- `selected_mask_id=conservative_seed_42`
- `features_count=12`
- `threshold=0.4`

Caption:

```text
Deployment readiness of the final binary IDS API, serving the P8 FedAvg + QGA model with the calibrated 12-feature QGA mask.
```

### Screenshot 3 — Online HTTP Replay Results

Files:

- `outputs/reports/p15_online_http_replay_summary.json`
- `outputs/reports/p15_online_http_replay_table.md`
- `outputs/reports/p15_online_http_replay_predictions.csv`

Capture:

- Rows replayed.
- Mean and p95 latency.
- Accuracy, attack recall, FPR, FNR, TP, TN, FP, FN if labels are available.
- Mention that the test holdout is used only for deployment replay evidence.

Caption:

```text
Online replay validation of held-out L1 flows through the final IDS API, measuring latency and binary IDS behavior without retraining or threshold tuning.
```

### Screenshot 4 — Grafana Online IDS Operational Overview

Recommended dashboard:

- `08_online_ids_operational_overview.json`

Grafana URL:

```text
http://127.0.0.1:3000
```

Panels to include:

- Final API readiness.
- Final API predictions/sec.
- Final API prediction errors/sec.
- Traffic generator published flows/sec.
- MQTT connection state.
- Final MQTT bridge readiness.
- Final MQTT bridge MQTT connection.
- Pipeline rates: bridge flows received, final predictions, final alerts.
- Online validator messages by family.

Caption:

```text
Operational overview of the online IDS deployment, showing service readiness, replay throughput, inference throughput, alerts, and MQTT topic observation.
```

### Screenshot 5 — Grafana IDS Security View

Recommended dashboard:

- `09_ids_security_view.json`

Panels to include:

- Final predictions total/rate.
- Final bridge predictions published/sec.
- Final bridge alerts published/sec.
- Runtime alert ratio.
- Bridge prediction errors.
- Final API HTTP latency p95 through the bridge.
- MQTT observed `ids/flows`, `ids/predictions`, and `ids/alerts` if the online validator is active.

Caption:

```text
Security-oriented IDS view showing final prediction publication, alert publication, alert ratio, bridge errors, and final API latency during safe replay validation.
```

### Screenshot 6 — Grafana Technical Observability

Recommended dashboard:

- `10_technical_observability.json`

Panels to include:

- Service up status by job.
- Final API readiness.
- Final MQTT bridge readiness.
- Final MQTT bridge MQTT connection.
- Final API error rate.
- Final bridge error rate.
- Traffic generator skipped rows.
- Mean and p95 final API latency through the bridge.
- Last final bridge message age.
- Online validator MQTT state.

Caption:

```text
Technical observability dashboard for validating service health, failure modes, MQTT observation, and inference latency during deployment replay.
```

### Screenshot 7 — MQTT Topic Observation

Files:

- `outputs/reports/p15_mqtt_topics_observed.json`
- `outputs/reports/p15_mqtt_topics_observed.csv`

Capture:

- `ids/flows/#` messages during traffic replay.
- `ids/predictions/#` and `ids/alerts/#` generated by `final-mqtt-bridge`.
- Explain that the final bridge connects MQTT replay to the final P8 FedAvg + QGA HTTP API.

Caption:

```text
MQTT topic observation during safe replay, distinguishing flow publication from prediction and alert publication.
```

### Screenshot 8 — Prometheus Targets

URL:

```text
http://127.0.0.1:9090/targets
```

Capture:

- `final-ids-api`
- `final-mqtt-bridge`
- `traffic-generator`
- Optional `online-validator`
- Any inactive optional profiles should be explained.

Caption:

```text
Prometheus target readiness for the final deployment validation stack.
```

## Recommended Dashboard Story for the Report

1. Start with P13: why P8 FedAvg + QGA is selected.
2. Show `final-ids-api /ready` and `/model/info`: model is packaged and loaded.
3. Run P15 HTTP replay: demonstrate online inference on held-out flows.
4. Show `final-mqtt-bridge /ready`: the MQTT bridge is connected and ready.
5. Show Prometheus targets: `final-ids-api` and `final-mqtt-bridge` are UP.
6. Show Grafana operational overview: replay, bridge, final API, predictions, and alerts are active.
7. Show Grafana security view: final prediction and alert publication.
8. Show technical observability: readiness, MQTT connectivity, latency, errors, and message freshness.

## Figure and Panel Quality Checklist

- Use a 5-15 minute time range while replay is active.
- Turn on auto-refresh at 5 seconds only during screenshots.
- Prefer rate panels over raw counters for live behavior.
- Use raw counters only for totals such as total flows, total alerts, or total predictions.
- Avoid panels whose metrics are not emitted in the active stack.
- Annotate screenshots to show that `final-mqtt-bridge` connects the historical MQTT replay path to the final HTTP API.
- Include at least one screenshot showing `P8 FedAvg + QGA`, `conservative_seed_42`, and `12 features`.
- Do not label Macro-F1, FPR, attack recall, or accuracy as live runtime metrics in Grafana screenshots.

## Limitations to State in the Report

- The final P8 model is deployed as an HTTP service.
- The historical MQTT replay path remains available, but final-model MQTT validation now uses `final-mqtt-bridge`.
- The test holdout is used only for deployment replay/evaluation, not for training, mask selection, or threshold tuning.
- P11 FedTN/MPS remains dry-run structural evidence and should not be shown as measured online accuracy.

## New Final Bridge Screenshot Checklist

- `http://127.0.0.1:8014/model/info`: final P8 FedAvg + QGA model, `conservative_seed_42`, 12 features.
- `http://127.0.0.1:8016/ready`: final MQTT bridge ready and connected.
- `http://127.0.0.1:9090/targets`: Prometheus targets with `final-mqtt-bridge` UP.
- Grafana Dashboard 08: replay + bridge + final API + predictions + alerts.
- Grafana Dashboard 09: final prediction and alert publication, alert ratio, bridge latency.
- Grafana Dashboard 10: technical observability, MQTT connection, errors, and last message age.
