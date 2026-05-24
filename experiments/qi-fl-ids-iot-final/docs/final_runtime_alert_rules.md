# Final Runtime Alert Rules

This project currently provisions Grafana dashboards and the Prometheus datasource, but it does not provision Grafana-managed alert rules. The rules below are therefore documented as report-ready and import-ready PromQL rules for the final online deployment validation stack.

These alerts monitor runtime behavior only. They must not be interpreted as offline IDS metrics such as Macro-F1, accuracy, attack recall, or FPR.

## Final IDS API

### FinalIDSApiDown

```promql
up{job="final-ids-api"} == 0
```

Meaning: Prometheus cannot scrape the final P8 FedAvg + QGA API.

### FinalIDSApiNotReady

```promql
final_ids_api_ready == 0
```

Meaning: the API process is alive, but the final model artifact is not ready for inference.

### FinalPredictionErrors

```promql
rate(final_ids_api_prediction_errors_total[5m]) > 0
```

Meaning: `/predict` requests are producing runtime errors.

## Final MQTT Bridge

### FinalMQTTBridgeDown

```promql
up{job="final-mqtt-bridge"} == 0
```

Meaning: Prometheus cannot scrape the final MQTT bridge.

### FinalMQTTBridgeNotReady

```promql
final_mqtt_bridge_ready == 0
```

Meaning: the bridge process is alive, but it is not ready to forward replayed flows.

### FinalMQTTBridgeDisconnected

```promql
final_mqtt_bridge_mqtt_connected == 0
```

Meaning: the bridge is not connected to Mosquitto, so MQTT replay messages cannot reach the final API path.

### FinalBridgePredictionErrors

```promql
rate(final_mqtt_bridge_prediction_errors_total[5m]) > 0
```

Meaning: the bridge is receiving malformed messages, cannot call the final API, or cannot publish the result.

### NoFinalPredictionsFromReplay

```promql
sum(rate(final_mqtt_bridge_flows_received_total[5m])) > 0
and
sum(rate(final_mqtt_bridge_predictions_published_total[5m])) == 0
```

Meaning: replayed flows are arriving at the bridge, but final predictions are not being published.

