# P16.6 Live MQTT Node-to-IDS Validation Evidence

## Objective

P16.6 freezes the runtime evidence that the live lab MQTT path works end to end for both prepared IoT nodes.

Validated path:

`VM node -> MQTT ids/flows/{node} -> final-mqtt-bridge -> final-ids-api -> ids/predictions/{node} / ids/alerts/{node}`

This report documents a technical pipeline validation. The zero-vector payloads used here are controlled JSON flow messages that validate routing, inference invocation, publication, and metrics. They are not a scientific performance evaluation; the scientific evaluation remains P12/P13.

## Architecture Validated

- Server IP: `192.168.56.1`
- MQTT broker: `192.168.56.1:1883`
- final-mqtt-bridge: `http://192.168.56.1:8016`
- final-ids-api: `http://192.168.56.1:8014`
- online-validator: `http://192.168.56.1:8015`
- live-lab-controller: `http://192.168.56.1:8020`
- Model: `p8_fedavg_qga_l1`
- Selected mask: `conservative_seed_42`

## VM1 Path

- Node: `iot-rpi-weak`
- Assigned tier: `weak`
- Input mode: `selected_12_scaled`
- Flow topic: `ids/flows/iot-rpi-weak`
- Prediction topic: `ids/predictions/iot-rpi-weak`
- Alert topic: `ids/alerts/iot-rpi-weak`

Observed metrics:

```text
final_mqtt_bridge_flows_received_total{node_id="iot-rpi-weak"} 1
final_mqtt_bridge_predictions_published_total{node_id="iot-rpi-weak",predicted_label="attack"} 1
final_mqtt_bridge_alerts_published_total{node_id="iot-rpi-weak",severity="critical"} 1
```

## VM2 Path

- Node: `iot-smart-watch-medium`
- Assigned tier: `medium`
- Input mode: `original_28_scaled`
- Flow topic: `ids/flows/iot-smart-watch-medium`
- Prediction topic: `ids/predictions/iot-smart-watch-medium`
- Alert topic: `ids/alerts/iot-smart-watch-medium`
- API behavior: final-ids-api accepts 28 scaled features and applies the final QGA mask server-side.

Observed MQTT topics:

```text
ids/flows/iot-smart-watch-medium observed
ids/predictions/iot-smart-watch-medium observed
ids/alerts/iot-smart-watch-medium observed
```

Observed metrics:

```text
final_mqtt_bridge_flows_received_total{node_id="iot-smart-watch-medium"} 1
final_mqtt_bridge_predictions_published_total{node_id="iot-smart-watch-medium",predicted_label="attack"} 1
final_mqtt_bridge_alerts_published_total{node_id="iot-smart-watch-medium",severity="critical"} 1
```

## Online Validator Evidence

Observed summary:

```text
ids/flows/iot-rpi-weak: 1
ids/predictions/iot-rpi-weak: 1
ids/alerts/iot-rpi-weak: 1
ids/flows/iot-smart-watch-medium: 1
ids/predictions/iot-smart-watch-medium: 1
ids/alerts/iot-smart-watch-medium: 1
```

## Error Counters

Observed error counters:

```text
final_ids_api_prediction_errors_total = 0
final_mqtt_bridge_prediction_errors_total = 0
```

## Interpretation

- The weak node path validates direct `selected_12_scaled` ingestion.
- The medium node path validates `original_28_scaled` ingestion with QGA mask application by final-ids-api.
- final-mqtt-bridge receives one flow and publishes one prediction plus one alert for each VM node.
- online-validator observes the expected flow, prediction, and alert topics for both nodes.
- No prediction errors were observed in final-ids-api or final-mqtt-bridge during the manual runtime validation.

## Limits

- P16.6 does not evaluate detection quality.
- P16.6 does not use live packet capture.
- P16.6 does not start any lab scenario.
- The next step is P16.7: real-time packet window feature extraction prototype.
