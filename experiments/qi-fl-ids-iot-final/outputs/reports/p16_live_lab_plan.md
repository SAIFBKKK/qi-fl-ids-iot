# P16 Live Lab Plan

Date: 2026-05-24

## Architecture

Server PC with Docker Desktop:

- `mosquitto` on `1883`.
- `final-ids-api` on `8014`.
- `final-mqtt-bridge` on `8016`.
- `online-validator` on `8015`.
- `dashboard-p13` on `8013`.
- `prometheus` on `9090`.
- `grafana` on `3000`.
- `live-lab-controller` on `8020` under compose profile `live-lab`.

VirtualBox VMs:

- VM 1: weak IoT node, e.g. `2 CPU / 2 GB RAM`.
- VM 2: medium IoT node, e.g. `4 CPU / 6 GB RAM`.
- Optional third VM or host-side script: controlled scenario simulator.

The network must be local and isolated. MQTT and controller ports are reachable
from the VMs; no external target is used.

## Docker Desktop and VirtualBox Choice

Docker Desktop keeps final inference, MQTT, monitoring, and validation
repeatable on the central server. Oracle VirtualBox gives visible, explainable
IoT VM nodes for the live demonstration without requiring physical devices.

## Mode 12 Selected Scaled Features

The VM agent publishes `input_mode=selected_12_scaled` and a 12-value `features`
array to `ids/flows/{node_id}`. The bridge forwards the list to
`final-ids-api /predict`; the API treats the vector as already masked/scaled.

## Mode 28 Scaled Features + QGA Mask

The VM agent or simulator publishes `input_mode=original_28_scaled` and a
28-value `features` array. The final API detects the 28-length input and applies
the QGA selected indices internally:

`[0, 2, 3, 4, 6, 13, 14, 19, 20, 25, 26, 27]`.

This demonstrates that the final API can support both the compact model input
and the fuller upstream feature pipeline shape.

## Prototype pcap-to-28-features

`deployment/live_feature_extractor` prepares:

`local controlled pcap -> passive packet reader -> flow/window grouping -> 28 feature alignment -> optional scaler -> IDS payload JSON/CSV`

The prototype is experimental. It reports unsupported or approximate fields and
does not modify the final model.

## MQTT Predictions and Alerts Validation

Flow:

1. VM agent or scenario simulator publishes `ids/flows/{node_id}`.
2. `final-mqtt-bridge` consumes `ids/flows/#`.
3. Bridge calls `final-ids-api /predict`.
4. Bridge publishes `ids/predictions/{node_id}`.
5. If predicted attack, bridge publishes `ids/alerts/{node_id}`.
6. VM agent prints alerts; `online-validator` collects topic evidence.

## Evidence Collector

Use:

- `online-validator /metrics` for MQTT message observation.
- `final-mqtt-bridge /metrics` for received flows, predictions, alerts, and
  bridge errors.
- `live-lab-controller /nodes` and `/assignments` for VM registration evidence.
- P16 validation reports from `src/scripts/16_validate_live_lab_setup.py`.

## Limits and Future Raw Live Extraction

P16 intentionally avoids real offensive traffic and live raw capture. Future
work can add a hardened raw packet collector only after safety review, exact
feature-semantic validation, and scaler packaging. The immediate live demo uses
controlled features, local replay files, or local pcaps.

