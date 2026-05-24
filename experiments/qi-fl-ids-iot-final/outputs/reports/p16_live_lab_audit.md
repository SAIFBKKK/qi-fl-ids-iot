# P16 Live Lab Audit

Date: 2026-05-24
Branch: `final/quantum-inspired-fl-iot-ids-final`

## Scope Audited

- `experiments/qi-fl-ids-iot-final/deployment/docker-compose.final.yml`
- `experiments/qi-fl-ids-iot-final/deployment/final_ids_api/`
- `experiments/qi-fl-ids-iot-final/deployment/final_mqtt_bridge/`
- `experiments/qi-fl-ids-iot-final/deployment/online_validator/`
- `experiments/qi-fl-ids-iot-final/deployment/l1_final/`
- `experiments/qi-fl-ids-iot-final/dashboard/`
- `services/traffic-generator/`
- `services/iot-node/`
- `services/edge-ids-gateway/`
- `services/mosquitto/`
- `services/monitoring/`
- `outputs/artifacts/features/feature_names.json`
- `outputs/artifacts/scalers/l1_binary_robust_scaler.pkl`
- `outputs/qga_feature_selection/final_selected_mask/`

## Scientific Reference

The P16 extraction prototype uses CICIoT2023/Sensors only as a conceptual
reference. CICIoT2023 publishes original packet captures and processed CSV
features; the CSV rows are extracted from packet windows between hosts. The
paper describes a DPKT-based extraction process and notes CICFlowMeter/NFStream
as alternative extraction tools. P16 does not reproduce attack execution; it
only prepares controlled feature replay and passive local pcap processing.

Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC10346235/

## 1. Services Already Ready for the Lab

- `mosquitto`: available in the final compose on port `1883`.
- `final-ids-api`: exposes P8 FedAvg + QGA inference on port `8014`.
- `final-mqtt-bridge`: subscribes to `ids/flows/#`, calls `/predict`, and
  publishes predictions/alerts.
- `online-validator`: observes `ids/flows/#`, `ids/predictions/#`,
  `ids/alerts/#`, and `ids/status/#`.
- `dashboard-p13`: available on port `8013` for final evidence display.
- `prometheus` and `grafana`: available on ports `9090` and `3000`.
- `traffic-generator`: historical controlled replay service under the `demo`
  profile.

## 2. Ports Accessible from the VMs

- `1883/tcp`: MQTT broker.
- `8020/tcp`: P16 live-lab-controller registration API.
- Optional read-only checks from the VM or presenter machine:
  - `8014/tcp`: final IDS API.
  - `8016/tcp`: final MQTT bridge health/readiness.
  - `8015/tcp`: online validator.
  - `8013/tcp`: dashboard.

## 3. MQTT Topics Used

- Input flows: `ids/flows/{node_id}` and bridge subscription `ids/flows/#`.
- Predictions: `ids/predictions/{node_id}`.
- Alerts: `ids/alerts/{node_id}`.
- Status/observation: `ids/status/{node_id}` and `ids/status/#`.

## 4. How VMs Publish Flows

Each VM runs `live_iot_node_agent/agent.py`, registers with
`http://<SERVER_IP>:8020/register-node`, receives its tier and topics, then
publishes controlled JSON payloads to `ids/flows/{node_id}`. Payloads include:
`flow_id`, `node_id`, `timestamp`, `input_mode`, `scenario`, and `features`.

## 5. How Alerts Are Received

`final-mqtt-bridge` republishes attack predictions to `ids/alerts/{node_id}`.
The VM agent subscribes to its assigned alert topic and prints received alerts
to the terminal. The `online-validator` can observe all alert topics for
evidence collection.

## 6. How to Test 12 Features vs 28 Features

- `selected_12_scaled`: the VM or simulator publishes exactly the 12 selected
  scaled features expected by the final model.
- `original_28_scaled`: the VM or simulator publishes the original 28 scaled
  L1 features. `final-ids-api` applies `selected_indices` from the QGA mask
  internally before inference.

The deployment schema confirms:

- `selected_mask_id`: `conservative_seed_42`.
- `original_feature_count`: `28`.
- `selected_feature_count`: `12`.
- `selected_indices`: `[0, 2, 3, 4, 6, 13, 14, 19, 20, 25, 26, 27]`.

## 7. Missing for raw packets -> 28 features -> scaler -> IDS

- A production-grade exact CICIoT2023-compatible extractor.
- A validated mapping from raw packet windows to the final 28 feature semantics.
- A scaler packaging decision for live deployment.
- Evidence comparing extractor output against known CSV rows from the same
  controlled pcap.
- Live capture safety controls; P16 remains passive/local and replay-only.

## 8. Out of Immediate Scope

- Running real attack tools or offensive traffic generation.
- Launching scans, floods, credential attempts, spoofing, or botnet emulation.
- Retraining or changing the P8 FedAvg + QGA model.
- Capturing live external traffic.
- Federated training orchestration during the live lab demo.

