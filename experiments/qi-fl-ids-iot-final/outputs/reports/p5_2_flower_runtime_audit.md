# P5.2 - True Flower Runtime Audit

## 1. Fichiers audités

### Anciennes expériences

- `experiments/fl-iot-ids-v3/src/fl/client_app.py`
- `experiments/fl-iot-ids-v3/src/fl/server_app.py`
- `experiments/fl-iot-ids-v3/src/fl/reporting_strategy.py`
- `experiments/fl-iot-ids-v3/src/fl/aggregation_hooks.py`
- `experiments/fl-iot-ids-v3/src/fl/metrics.py`
- `experiments/fl-iot-ids-v3/src/scripts/run_server.py`
- `experiments/fl-iot-ids-v3/src/scripts/run_client.py`
- `experiments/fl-iot-ids-v3/src/scripts/run_experiment.py`
- `experiments/fl-iot-ids-v3/configs/fl/fedavg_30rounds.yaml`
- `experiments/fl-iot-ids-v3/pyproject.toml`

### Services et orchestration

- `services/fl-server/server_entrypoint.py`
- `services/fl-server/real_runner_wrapper.py`
- `services/fl-server/requirements.txt`
- `services/fl-client/client_entrypoint.py`
- `services/fl-client/requirements.txt`
- `services/iot-node/*`
- `services/docker-compose.yml`
- `.github/workflows/ci.yml`

### Documentation Flower consultée

- Official Flower PyTorch quickstart: `ClientApp`, `ServerApp`, `task.py`, `pyproject.toml`, `flwr run . --stream`.
- Official Flower simulation docs: local simulations are the default Flower app runtime in current Flower releases.

Local runtime check:

- Installed local Flower version: `flwr 1.8.0`.
- Available APIs: `ClientApp=True`, `ServerApp=True`, `run_simulation=True`, `start_server=True`, `start_client=True`, `start_numpy_client=True`.

## 2. Pipeline FL v3 trouvé

The v3 experiment has two Flower paths:

- Modern-ish Flower simulation path:
  - `src/fl/server_app.py` builds `ServerApp` with `ReportingFedAvg`, QIFA, QIFA-guard or SCAFFOLD strategies.
  - `src/fl/client_app.py` builds `ClientApp` wrapping a `NumPyClient`.
  - `src/scripts/run_experiment.py` calls `flwr.simulation.run_simulation(server_app=..., client_app=..., num_supernodes=...)`.
  - `pyproject.toml` declares Flower app components for newer `flwr run` workflows.
- Legacy path:
  - `src/scripts/run_server.py` uses `flwr.server.start_server`.
  - `src/scripts/run_client.py` uses `flwr.client.start_client`.

The v3 logs came from Python logging inside clients, strategy callbacks, and Flower's simulation/runtime logs. Metrics were generated via strategy aggregation hooks and artifact trackers.

## 3. Pipeline FL v2/moderne trouvé

The repository CI still tests `experiments/fl-iot-ids-v2`, but the active modern FL implementation is v3. The services layer also contains a `fl-server`/`fl-client` training profile:

- `TRAINING_MODE=mock` starts a real Flower server and mock clients for orchestration validation only.
- `TRAINING_MODE=real` launches the v3 scientific runner from `fl-server`; the Docker `fl-client` containers exit cleanly in real mode.

This means the service layer is useful for future P14 orchestration, but not the right source for scientific P5.2 metrics.

## 4. Ancien style Flower vs nouveau style Flower

| Aspect | Ancien style | Nouveau style |
| --- | --- | --- |
| Server | `flwr.server.start_server` | `ServerApp` |
| Client | `start_client` / `start_numpy_client` | `ClientApp` |
| Local execution | multiple terminal processes | `run_simulation` or `flwr run . --stream` |
| Current repo usage | v3 `run_server.py` / `run_client.py`, services mock | v3 `run_experiment.py`, `server_app.py`, `client_app.py` |
| P5.2 choice | fallback only | selected path |

Because local `flwr 1.8.0` supports `ClientApp`, `ServerApp`, and `run_simulation`, P5.2 should use the modern app structure while remaining compatible with the older 1.8 constructor signatures.

## 5. Éléments réutilisables

- v3 `ClientApp`/`ServerApp` structure and simulation pattern.
- v3 strategy callback pattern for fit/evaluate aggregation.
- v3 communication accounting idea: model payload bytes multiplied by active clients and upload/download directions.
- P5 in-process L1 model, metrics, threshold tuning, scenario loading and console logging helpers.
- P3 partition manifests as the source of client train/val counts.
- P4 metrics as baseline comparison input.

## 6. Éléments à éviter

- Reusing v3 34-class model and class-weight logic for P5.2 L1.
- Reusing QIFA, QIFA-guard, SCAFFOLD or multi-tier code in P5.2.
- Using Docker services for this phase.
- Sending the global test holdout to clients.
- Depending on current Flower 1.29-only app APIs while local runtime is 1.8.0.

## 7. Risques

- Flower import is slower than normal on this workspace, so smoke should remain one round and sampled.
- Flower 1.8 `ClientApp`/`ServerApp` signatures differ from current docs; implementation must be compatible with 1.8.
- `flwr run . --stream` is the current recommended CLI, but local compatibility is safer through `flwr.simulation.run_simulation`.
- Ray/simulation runtime may be unavailable or heavy on some machines; verify must detect this before smoke.
- Flower simulation still runs clients in one local runtime, but it is a true Flower runtime path with real Flower client/server apps and strategy callbacks.

## 8. Recommandation d’architecture P5.2

Use a separate final package:

- `src/fl_l1_flower/task.py` for PyTorch model, train/eval, and parameter conversion.
- `src/fl_l1_flower/client_app.py` for Flower `NumPyClient` and `ClientApp`.
- `src/fl_l1_flower/server_app.py` for Flower `ServerApp` and `run_simulation`.
- `src/fl_l1_flower/strategy.py` for `FlowerL1FedAvgStrategy`.
- `src/fl_l1_flower/report_builder.py` for P5.2 docs and summaries.

Execution choice:

- Primary: Flower `ClientApp`/`ServerApp` with `flwr.simulation.run_simulation`, compatible with installed `flwr 1.8.0`.
- Smoke/full fallback: Flower legacy localhost runtime with `flwr.server.start_server` and `flwr.client.start_client`.
- Documented future upgrade: package app for `flwr run . --stream` when the environment uses the newer Flower app CLI consistently.

P5.2 smoke note: the first `run_simulation` attempt on Windows/Ray was interrupted and left a Ray cluster running. The cluster was stopped with `ray stop --force`. The reliable smoke path is therefore the true Flower legacy localhost runtime, while the ClientApp/ServerApp modules remain implemented and importable.
