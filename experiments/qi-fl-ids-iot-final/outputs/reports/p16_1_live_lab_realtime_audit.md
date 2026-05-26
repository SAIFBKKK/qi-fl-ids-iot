# P16.1 Live Lab Real-Time IDS Audit

Date: 2026-05-26
Branch: `final/quantum-inspired-fl-iot-ids-final`

## Scope

Audited for P16.1 step 0:

- `deployment/docker-compose.final.yml`
- `deployment/final_ids_api/`
- `deployment/final_mqtt_bridge/`
- `deployment/online_validator/`
- `deployment/live_lab_controller/`
- `deployment/live_iot_node_agent/`
- `deployment/live_attack_simulator/`
- `deployment/live_feature_extractor/`
- `outputs/artifacts/features/feature_names.json`
- `outputs/qga_feature_selection/final_selected_mask/feature_mask.json`
- `outputs/artifacts/scalers/l1_binary_robust_scaler.pkl`

## Existing Server Stack

`docker-compose.final.yml` already provides the services needed for the future
server PC demo:

- `mosquitto` on port `1883`.
- `final-ids-api` on port `8014`.
- `final-mqtt-bridge` on port `8016` under profile `online`.
- `online-validator` on port `8015` under profile `online`.
- `live-lab-controller` on port `8020` under profile `live-lab`.
- `dashboard-p13` on port `8013`.
- `prometheus` on port `9090`.
- `grafana` on port `3000`.

No Docker services are launched in P16.1 step 0.

## Final IDS API

`final_ids_api` exposes:

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /model/info`
- `POST /predict`
- `POST /predict/batch`

It accepts:

- 12 selected scaled features;
- 28 original scaled features, followed by QGA mask application inside the API.

The final selected mask is `conservative_seed_42`.

## MQTT Bridge and Validator

`final_mqtt_bridge` subscribes to `ids/flows/#`, converts MQTT payloads into
HTTP `/predict` requests, then republishes:

- predictions to `ids/predictions/{node_id}`;
- alerts to `ids/alerts/{node_id}`.

`online_validator` observes:

- `ids/flows/#`
- `ids/predictions/#`
- `ids/alerts/#`
- `ids/status/#`

This is already aligned with VM replay and future real-time extraction.

## Live Lab Controller and Node Agent

`live_lab_controller` already supports node registration and assigns tiers:

- weak;
- medium;
- powerful.

`live_iot_node_agent` already prepares node registration, controlled flow
publishing, and alert subscription. P16.1 adds a separate clonable
`experiments/live_lab/` directory for VM-side work without changing the P16
deployment services.

## Feature Replay and Feature Extractor

`live_attack_simulator` is controlled feature replay only. It does not run live
network actions.

`live_feature_extractor` is prepared as a safe local pcap-to-feature prototype.
The future real-time path still needs a runtime packet-window agent and a clear
scaler/model packaging decision for edge inference.

## Feature Schema and QGA Mask

The final 28-feature order is available in:

`outputs/artifacts/features/feature_names.json`

The feature list is:

1. `flow_duration`
2. `Header_Length`
3. `Protocol Type`
4. `Duration`
5. `Rate`
6. `fin_flag_number`
7. `syn_flag_number`
8. `rst_flag_number`
9. `psh_flag_number`
10. `ack_flag_number`
11. `ack_count`
12. `syn_count`
13. `fin_count`
14. `urg_count`
15. `rst_count`
16. `HTTP`
17. `HTTPS`
18. `DNS`
19. `SSH`
20. `TCP`
21. `UDP`
22. `ARP`
23. `ICMP`
24. `Tot sum`
25. `Min`
26. `Std`
27. `IAT`
28. `Number`

The final QGA mask is:

- `mask_id`: `conservative_seed_42`
- selected indices: `[0, 2, 3, 4, 6, 13, 14, 19, 20, 25, 26, 27]`
- selected feature count: `12`

## Scaler Availability

The local robust scaler exists at:

`outputs/artifacts/scalers/l1_binary_robust_scaler.pkl`

It is not part of the lightweight final deployment bundle. Future real-time
edge inference must decide whether to package this scaler or keep all inference
server-side.

## Ports Needed by the VMs

Required:

- `1883`: MQTT broker.
- `8020`: live-lab-controller registration.

Optional validation/evidence:

- `8014`: final-ids-api readiness.
- `8016`: final-mqtt-bridge readiness.
- `8015`: online-validator.
- `8013`: dashboard.
- `9090`: Prometheus.
- `3000`: Grafana.

## MQTT Topics

- Publish flows: `ids/flows/{node_id}`
- Receive predictions: `ids/predictions/{node_id}`
- Receive alerts: `ids/alerts/{node_id}`
- Status/evidence: `ids/status/{node_id}` and `ids/status/#`

## What Already Exists

- Final API inference for 12 and 28 scaled features.
- MQTT bridge from `ids/flows/#` to API predictions.
- Validator and monitoring endpoints.
- Live lab registration service.
- Controlled feature replay components.
- Static pcap-to-feature prototype.

## Missing for Real-Time Extraction

- VM-side packet-window runtime that is safe, configurable, and disabled by
  default.
- Exact mapping from real packet windows to all 28 project features.
- Runtime scaler packaging for VM-side edge inference.
- Medium-node edge model packaging and compatibility checks.
- Evidence workflow for screenshots, Grafana panels, and MQTT traces.
- VirtualBox VM creation on `E:\VirtualBox VMs\qi-fl-ids-iot-live-lab\`.

## Step 0 Boundary

P16.1 step 0 creates structure and validation only. It does not launch Docker,
create VMs, capture packets, generate live traffic, train models, run Flower, or
modify P8-P16 results.

