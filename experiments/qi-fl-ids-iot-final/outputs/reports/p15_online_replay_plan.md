# P15 Online Replay and Operational Deployment Validation Plan

Date: 2026-05-21

## Objective

Validate the final L1 deployment operationally by replaying held-out flows through the P14 stack without training, Flower, mask selection, or threshold tuning.

The held-out data is used only as an online deployment simulator and final measurement source.

## P15-A: HTTP Replay to final-ids-api

Create `15_run_online_http_replay.py`.

Responsibilities:

- Load `test_scaled.npz` or `deployment_15`.
- Send rows to `final-ids-api /predict`.
- Support 12-feature QGA selected mode and 28-feature scaled mode.
- Measure latency per request.
- If labels are available, compute online accuracy, attack recall, FPR, FNR, TP, TN, FP, FN.
- Write:
  - `p15_online_http_replay_summary.json`
  - `p15_online_http_replay_predictions.csv`
  - `p15_online_http_replay_table.md`

Default operational validation command:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/15_run_online_http_replay.py --api-url http://127.0.0.1:8014 --max-rows 1000 --sleep-ms 0 --use-qga-mask
```

## P15-B: MQTT Replay Observation

Create an optional `online-validator` Docker service.

Responsibilities:

- Subscribe to:
  - `ids/flows/#`
  - `ids/predictions/#`
  - `ids/alerts/#`
  - `ids/status/#`
- Count messages by topic family.
- Track latest message timestamps and optional flow-to-prediction latency when `flow_id` and timestamps are available.
- Expose:
  - `GET /health`
  - `GET /ready`
  - `GET /metrics`
  - `GET /summary`

Add the service to `docker-compose.final.yml` under profile `online`, port `8015`.

## P15-C: Online Validator and MQTT Topic Check

Create `15_check_mqtt_topics.py`.

Responsibilities:

- Connect to Mosquitto.
- Subscribe to `ids/#`.
- Observe messages for a bounded duration.
- Write:
  - `p15_mqtt_topics_observed.json`
  - `p15_mqtt_topics_observed.csv`

This validates whether traffic replay is active and whether predictions/alerts are actually published.

## P15-D: Evidence Collection

Create `15_collect_online_evidence.py`.

Responsibilities:

- Query:
  - final-ids-api `/health`, `/ready`, `/metrics`
  - dashboard `/health`
  - traffic-generator `/health` if active
  - online-validator `/health`, `/ready`, `/metrics` if active
- Write:
  - `p15_online_evidence.json`
  - `p15_online_evidence_table.md`

## P15-E: Future Live Lab Adapter

Future work:

- Add a final MQTT inference bridge that consumes `ids/flows/#`, calls `final-ids-api`, and publishes P8/P8+QGA binary predictions to `ids/predictions/#` and alerts to `ids/alerts/#`.
- Optionally adapt `edge-ids-gateway` to the final P8 binary deployment bundle and scaled feature contract.
- Add Prometheus scrape target for `online-validator`.

## Constraints

- No training.
- No Flower runtime.
- No modification to P8-P14 results.
- Test holdout is not used for model choice, threshold tuning, or QGA selection.
- Heavy datasets, runs, logs, and checkpoints are not committed.

## Acceptance Evidence

- Audit and plan generated.
- HTTP replay script exists and can write reports.
- MQTT observer service exists and is optional in compose.
- MQTT topic check script exists.
- Evidence collector exists.
- Integration tests pass without requiring long-running Docker services.
