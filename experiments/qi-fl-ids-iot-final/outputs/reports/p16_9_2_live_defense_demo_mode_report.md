# P16.9.2 Live Defense Demo Mode Report

## Objective

P16.9.2 adds a live presentation mode for the final jury defense. The new page guides the demonstration from platform readiness to live MQTT-to-IDS alerts without changing the scientific results or launching unsafe activity.

## Dashboard Architecture

- Existing dashboard service: `dashboard-p13` on port `8013`.
- New route: `/demo`.
- New state endpoint: `/api/live-lab/demo-state`.
- Existing aggregated source: `/api/live-lab/state`.
- Runtime sources:
  - live-lab-controller `/health`, `/ready`, `/nodes`, `/assignments`
  - online-validator `/summary`
  - final-mqtt-bridge `/ready`, `/metrics`
  - final-ids-api `/ready`, `/metrics`, optional `/model/info`

## Demo State Endpoint

`GET /api/live-lab/demo-state` returns:

- `platform_status`
- boolean `steps`
- display-ready `step_details`
- live `devices`
- controller `assignments`
- `model_profile`
- `latest_alert`
- `metrics`
- `recent_events`
- `services`
- `warnings`

The endpoint is defensive. If one service is unavailable, the response still renders with clear warnings.

## Panels Added

- Big live defense header with platform status and last refresh.
- Demo Progress Timeline:
  - Platform Ready
  - Devices Connected
  - Model Assigned
  - Packet Window Generated
  - MQTT Flow Received
  - IDS Prediction Produced
  - Alert Detected
  - Zero Runtime Errors
- Live Device Cards for `iot-rpi-weak` and `iot-smart-watch-medium`.
- Model Assignment Panel for P8 FedAvg + QGA.
- Live Alert Focus Panel with severity, label, confidence, flow ID, timestamp, and source topic.
- Live Metrics Panel for VM1/VM2 flows, predictions, alerts, and runtime errors.
- Live Event Stream for device, model, flow, prediction, and alert events.
- Jury Explanation Panel.
- Presentation Controls with safe navigation and refresh actions only.

## Step Logic

- `platform_ready`: controller, validator, bridge, and API are reachable.
- `devices_connected`: both expected VM nodes are registered.
- `model_assigned`: both nodes have the expected final model, mask, and tier.
- `packet_window_generated`: at least one controlled PacketWindow flow is observed.
- `mqtt_flows_seen`: bridge or validator counters show flows.
- `predictions_seen`: bridge or validator counters show predictions.
- `alerts_seen`: bridge or validator counters show alerts.
- `zero_errors`: final IDS API and MQTT bridge prediction error counters are zero.

## Safety Scope

P16.9.2 is a dashboard and runbook phase. It does not run capture, lab attack scenarios, training, Flower, or model tuning. The live evidence path uses controlled PacketWindow(30) JSON publication only.

## Limits

- The demo mode validates deployment behavior, not scientific model performance.
- Scientific performance remains documented in P12/P13.
- The page relies on polling and in-memory browser notification state.
- Optional `/model/info` is used when available; otherwise the dashboard displays final model defaults.
