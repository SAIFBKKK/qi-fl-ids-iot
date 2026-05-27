# P16.9 Live Lab Dashboard UX Plan

## Objective

P16.9 upgrades `dashboard-p13` into a professional live lab monitoring and security dashboard for the final defense demo.

## Architecture

The existing FastAPI dashboard remains the single dashboard service on port `8013`.

New backend endpoint:

`GET /api/live-lab/state`

This endpoint aggregates the live lab state from the existing services and returns one compact JSON document for the frontend.

## Data Sources

- live-lab-controller:
  - `/health`
  - `/ready`
  - `/nodes`
  - `/assignments`
- online-validator:
  - `/health`
  - `/summary`
- final-mqtt-bridge:
  - `/ready`
  - `/metrics`
- final-ids-api:
  - `/ready`
  - `/metrics`

The dashboard tries Docker service names first and localhost fallbacks second, so it works inside the compose network and during local development.

## Notification Logic

The frontend polls `/api/live-lab/state` every few seconds.

Device notifications:

- Keep a browser-side set of known `node_id` values.
- Show a toast when a node appears for the first time in the current page session.
- Store the event in the visible "Node Arrival Stream" history.

Alert notifications:

- Read recent alert samples from online-validator.
- Build a stable alert key from topic, flow id, timestamp, and receive time.
- Show a colored toast only for unseen alerts.
- Render the chronological "Recent Alerts / Security Events" panel.

This prevents repeated notifications when no data changes.

## Light And Dark Mode

The dashboard includes a visible theme toggle.

- Theme state is stored in `localStorage`.
- Both themes use the same components and semantic alert colors.
- Cards, tables, badges, toasts, and alert panels use CSS custom properties.

## Dashboard Panels

- Overview KPI cards: devices, flows, predictions, alerts, API errors, bridge errors.
- Device connected notification area.
- Device and model assignment table.
- Model profile panel for `p8_fedavg_qga_l1` and `conservative_seed_42`.
- Recent alerts / security events panel.
- Service status panel for controller, validator, bridge, and final IDS API.
- MQTT topic counts evidence panel.

## Limits

- P16.9 does not change model artifacts or scientific results.
- P16.9 does not start packet capture, training, Flower, or lab scenarios.
- The dashboard relies on polling rather than SSE to keep the runtime simple and robust.
