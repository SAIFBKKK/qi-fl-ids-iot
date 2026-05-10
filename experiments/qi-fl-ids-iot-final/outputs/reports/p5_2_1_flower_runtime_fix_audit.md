# P5.2.1 - Flower Runtime Fix Audit

## 1. Scope

This audit targets the P5.2 true Flower runtime under:

- `src/fl_l1_flower/legacy_server.py`
- `src/fl_l1_flower/legacy_client.py`
- `src/scripts/05_2_run_flower_l1_smoke.py`

No P5 in-process code, Docker, dashboard, QI module, or legacy experiment was modified.

## 2. Observed blockage

The successful smoke used `legacy-local`, but a full `legacy-local` run blocked after client data loading:

- server started;
- full clients loaded all P3 train/val samples;
- no fresh `fit started | round=1` appeared;
- the Python process stayed alive;
- old smoke and full logs were mixed in the same scenario-level log files.

## 3. Findings

### Server/client launch model

Before P5.2.1, `run_legacy_local_smoke` launched the Flower server and all clients as Python threads inside the same process. This can work for a tiny smoke run, but it is fragile for full-sized clients on Windows because the Flower gRPC runtime, logging, and client data loading all share the same process.

### Address and min clients

The default address was `127.0.0.1:8095` in the launcher. The requested manual mode uses `127.0.0.1:8080`.

The Flower config has:

- `min_fit_clients=3`
- `min_evaluate_clients=3`
- `min_available_clients=3`

These match `K=3`, so the server correctly waits for all three clients.

### Client start path

Clients did call `start_client` after construction in the code path, but there were no explicit logs before/after connection. If a client blocked during connection or Flower handshake, the logs looked like data loading had completed but nothing else happened.

### Exception visibility

Thread-mode exceptions were collected in a shared list and raised only after joins/timeouts. They were not written immediately to per-client log files, making the failure mode hard to diagnose.

### Log mixing

Before P5.2.1, `flower_clients.log`, `flower_server.log`, and `run_console.log` lived directly under `alpha_0.5/k3/logs`. Smoke/full executions reused the same files and could mix old and new lines.

## 4. Cause probable

The most likely cause is the thread-based `legacy-local` launcher: server and clients share one Python process on Windows while full-sized client arrays are loaded in memory and Flower gRPC waits for all clients. This makes the runtime less reliable and less demonstrative than real separate processes.

## 5. Fix selected

P5.2.1 introduces:

- one `run_id` per execution: `run_YYYYMMDD_HHMMSS`;
- isolated logs/artifacts under `outputs/fl_l1_flower/alpha_0.5/k3/runs/{run_id}/`;
- `latest_run.json` at scenario level;
- explicit server/client logs;
- manual server/client scripts;
- auto-subprocess launcher using separate Python processes;
- port availability check before starting a server;
- test holdout protection preserved: clients load only train/val, server alone may load global test after validation.

## 6. Manual runtime recommendation

Use manual mode for long/full P5.2 runs:

1. Start server.
2. Start `client_1`.
3. Start `client_2`.
4. Start `client_3`.

This gives real Flower logs and isolates each client process for Windows debugging.

## 7. Acceptance status

P5.2.1 is accepted when compile, tests, verify, and a one-round subprocess smoke pass. Full 30-round Flower is intentionally not launched by Codex.
