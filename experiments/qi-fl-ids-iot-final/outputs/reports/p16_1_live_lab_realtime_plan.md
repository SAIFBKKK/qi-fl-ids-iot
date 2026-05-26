# P16.1 Live Lab Real-Time IDS Plan

Date: 2026-05-26

## Step 0: Git Structure and Audit

- Create `experiments/live_lab/` as a clonable VM workspace.
- Add config examples for server and nodes.
- Add dry-run node placeholders.
- Add future real-time agent skeletons.
- Add validation script and integration tests.
- Do not start Docker, create VMs, capture packets, or change final model
  outputs.

## Step 1: Server Preparation

- Use Docker Desktop on the central PC.
- Later start only the needed compose profiles:
  - base stack for MQTT/API/dashboard/monitoring;
  - `online` for bridge and validator;
  - `live-lab` for the controller.
- Verify ports `1883`, `8014`, `8015`, `8016`, and `8020`.

## Step 2: VM Preparation on E: with Oracle VirtualBox

- Create VMs later under:
  `E:\VirtualBox VMs\qi-fl-ids-iot-live-lab\`
- VM1: `iot-rpi-weak`, raspberry-like weak node.
- VM2: `iot-smart-watch-medium`, smart-watch-like medium node.
- Keep the network local and isolated.

## Step 3: Clone Repo in VMs

- Clone the repository or copy `experiments/live_lab/`.
- Start with dry-run scripts only.
- Confirm configs in `experiments/live_lab/configs/`.

## Step 4: Node Registration

- Weak node registers as `iot-rpi-weak`.
- Medium node registers as `iot-smart-watch-medium`.
- Both use `live-lab-controller` on port `8020`.
- Controller returns MQTT publish, prediction, and alert topics.

## Step 5: Feature Replay 12/28

- Weak node first uses `selected_12_scaled`.
- Medium node first uses `original_28_scaled`.
- The final API applies the QGA mask for 28-feature scaled payloads.
- Replay remains controlled JSON feature replay, not live network activity.

## Step 6: Real-Time Packet Window Extraction

- Add a disabled-by-default packet-window runtime.
- Convert packet windows to a 28-feature row.
- Preserve exact feature order from `feature_names.json`.
- Mark unsupported features explicitly.
- Apply scaler only when packaged and validated.

## Step 7: Edge Inference and Server-Side Inference

- Weak node: server-side inference through MQTT bridge and final API.
- Medium node: future edge inference placeholder, then optional server-side
  comparison.
- Do not change the final P8 FedAvg + QGA model.

## Step 8: MQTT Predictions and Alerts Validation

- Observe `ids/flows/{node_id}`.
- Confirm `ids/predictions/{node_id}`.
- Confirm `ids/alerts/{node_id}` when a controlled payload is classified as an
  alert.
- Use `online-validator` and bridge metrics for evidence.

## Step 9: Dashboard and Grafana Evidence

- Use dashboard P13 for final model/evidence context.
- Use Grafana/Prometheus for runtime metrics.
- Collect controller registrations and MQTT bridge counters.

## Step 10: Screenshots and Final Evidence

- Capture server health endpoints.
- Capture VM dry-run and registration terminals.
- Capture MQTT prediction/alert evidence.
- Capture dashboard and Grafana panels.
- Write a final evidence report without adding logs, pcap files, datasets, or
  checkpoints.

