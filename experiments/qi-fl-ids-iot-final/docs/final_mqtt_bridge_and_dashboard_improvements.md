# Final MQTT Bridge and Dashboard Improvements

## Previous Gap

The project had two deployment paths:

- The final scientific deployment path: `final-ids-api`, serving the final P8 FedAvg + QGA L1 binary IDS model through HTTP.
- The historical MQTT operational path: `traffic-generator -> ids/flows/{node} -> iot-node -> ids/predictions/{node} / ids/alerts/{node}`.

The historical path was observable in Prometheus and Grafana, but it did not use the final P8 FedAvg + QGA model. The final HTTP API used the final model, but replayed MQTT flows did not pass through it.

## New Final MQTT Bridge

P15 adds `final-mqtt-bridge` as the missing operational link:

```text
traffic-generator
    -> MQTT ids/flows/{node}
    -> final-mqtt-bridge
    -> final-ids-api /predict
    -> MQTT ids/predictions/{node}
    -> MQTT ids/alerts/{node}
    -> Prometheus
    -> Grafana
```

The bridge subscribes to `ids/flows/#`, extracts `node_id`, `flow_id`, and scaled features, calls `final-ids-api /predict`, and republishes the final prediction and optional alert to MQTT.

This makes the final P8 FedAvg + QGA model observable through the same operational tooling as the MQTT replay stack.

## MQTT Topics

| Topic | Producer | Consumer | Purpose |
|---|---|---|---|
| `ids/flows/{node}` | `traffic-generator` or safe validation script | `final-mqtt-bridge`, `online-validator` | Replayed flow events |
| `ids/predictions/{node}` | `final-mqtt-bridge` | `online-validator`, Grafana via Prometheus | Final model prediction events |
| `ids/alerts/{node}` | `final-mqtt-bridge` | `online-validator`, Grafana via Prometheus | Alert events when the final model predicts attack |

## Final API Role

`final-ids-api` remains the source of truth for inference. It loads the final L1 deployment bundle:

- Model: P8 FedAvg + QGA.
- Mask: `conservative_seed_42`.
- Input dimension: 12 selected scaled features.
- Threshold: 0.4.

The bridge does not train, tune, select a threshold, or change the model. It only forwards replayed feature vectors to `/predict`.

## Prometheus Metrics

The bridge exposes:

- `final_mqtt_bridge_ready`
- `final_mqtt_bridge_mqtt_connected`
- `final_mqtt_bridge_flows_received_total`
- `final_mqtt_bridge_predictions_published_total`
- `final_mqtt_bridge_alerts_published_total`
- `final_mqtt_bridge_prediction_errors_total`
- `final_mqtt_bridge_http_latency_seconds`
- `final_mqtt_bridge_last_message_timestamp_seconds`

These metrics are runtime metrics. They measure deployment behavior, not offline scientific performance.

## Grafana Dashboard Improvements

Three runtime dashboards were updated.

### Dashboard 08 - Online IDS Operational Overview

Purpose: one screenshot showing that replay, the MQTT bridge, the final API, predictions, and alerts are active.

Key panels:

- Final API readiness.
- Final API predictions/sec.
- Final API prediction errors/sec.
- Traffic generator status.
- Traffic generator MQTT connection.
- Traffic generator published flows/sec.
- Final MQTT bridge readiness.
- Final MQTT bridge MQTT connection.
- Final MQTT bridge received flows/sec.
- Final MQTT bridge published predictions/sec.
- Final MQTT bridge published alerts/sec.
- Online validator observed messages/sec.

### Dashboard 09 - IDS Security View

Purpose: show what the final IDS publishes during replay without mixing in offline evaluation metrics.

Key panels:

- Final predictions total/rate.
- Final bridge predictions published/sec.
- Final bridge alerts published/sec.
- Runtime alert ratio.
- Prediction errors.
- Final API HTTP latency p95 through the bridge.
- MQTT observed `ids/flows`, `ids/predictions`, and `ids/alerts` if `online-validator` is active.

### Dashboard 10 - Technical Observability

Purpose: show service health and failure modes.

Key panels:

- Service `up` by job.
- Final API readiness.
- Final MQTT bridge readiness.
- Final MQTT bridge MQTT connection.
- Final API prediction error rate.
- Final bridge prediction error rate.
- Final bridge HTTP latency mean/p95.
- Traffic generator skipped rows.
- Online-validator MQTT connection.
- Final bridge last message age.

## Screenshot Plan

Recommended screenshots for the final report:

1. `final-ids-api /model/info`, showing P8 FedAvg + QGA, `conservative_seed_42`, 12 features, and threshold 0.4.
2. `final-mqtt-bridge /ready`, showing MQTT connection and bridge readiness.
3. Prometheus targets page with `final-ids-api`, `final-mqtt-bridge`, `traffic-generator`, and `online-validator` UP.
4. Dashboard 08 showing replay throughput, bridge throughput, final API readiness, and alert publication.
5. Dashboard 09 showing final predictions, final alerts, alert ratio, errors, and latency.
6. Dashboard 10 showing service health, MQTT connection state, error rates, and last message age.

## Safe Validation Commands

Start the final online stack:

```powershell
cd C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\deployment
docker compose -f docker-compose.final.yml --profile online --profile demo up -d --build
```

Validate endpoints only:

```powershell
python C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\src\scripts\15_validate_final_mqtt_bridge.py
```

Publish one safe synthetic scaled-zero replay payload:

```powershell
python C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\src\scripts\15_validate_final_mqtt_bridge.py --publish-sample
```

## Limitations

- The traffic generator still uses historical replay payloads. The bridge accepts selected 12-feature vectors, original 28-feature vectors, or feature-name dictionaries when the schema is available.
- The online dashboards are runtime dashboards only. They intentionally do not show Macro-F1, accuracy, FPR, attack recall, or FL rounds.
- Offline scientific metrics remain in P12/P13 reports and should be cited separately from runtime replay screenshots.
- Grafana alert rules are documented in `final_runtime_alert_rules.md` because this repository currently provisions dashboards and datasources, not Grafana alerting rules.

## Report-Ready Paragraph

The final online deployment was improved by adding a bridge between the MQTT replay layer and the final IDS HTTP API. This bridge subscribes to replayed flow topics, forwards feature vectors to the P8 FedAvg + QGA model, and republishes final predictions and alerts through MQTT. As a result, the final scientific model becomes observable through Prometheus and Grafana while preserving the separation between offline evaluation metrics and runtime deployment metrics.

