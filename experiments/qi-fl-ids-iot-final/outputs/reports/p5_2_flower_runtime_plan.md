# P5.2 - True Flower Runtime Plan

## 1. Architecture choisie

Selected architecture: **Option 1 - Flower ClientApp/ServerApp**, adapted to installed `flwr 1.8.0`, with **Option 2 legacy localhost fallback** for reliable smoke/full execution on this Windows workspace.

Rationale:

- Local runtime exposes `ClientApp`, `ServerApp`, and `flwr.simulation.run_simulation`.
- v3 already validates this pattern in `src/scripts/run_experiment.py`.
- Current official Flower docs recommend `ClientApp`, `ServerApp`, `task.py`, `pyproject.toml`, and `flwr run . --stream`, but local 1.8 signatures are older. P5.2 will therefore implement the same app concepts and run them through `run_simulation`.

Legacy `start_server/start_client` is implemented because the first Ray-backed `run_simulation` smoke attempt was interrupted and did not complete promptly on Windows. The fallback is still a true Flower runtime: it starts a real Flower server and real Flower clients over localhost.

## 2. Fichiers à créer

- `configs/fl_l1_flower.yaml`
- `src/fl_l1_flower/__init__.py`
- `src/fl_l1_flower/task.py`
- `src/fl_l1_flower/client_app.py`
- `src/fl_l1_flower/server_app.py`
- `src/fl_l1_flower/strategy.py`
- `src/fl_l1_flower/metrics.py`
- `src/fl_l1_flower/data.py`
- `src/fl_l1_flower/communication.py`
- `src/fl_l1_flower/report_builder.py`
- `src/fl_l1_flower/verify_flower_setup.py`
- `src/scripts/05_2_verify_flower_l1_setup.py`
- `src/scripts/05_2_run_flower_l1_smoke.py`
- `docs/05_2_flower_runtime.md`
- `tests/unit/test_flower_l1_strategy.py`
- `tests/integration/test_flower_l1_setup.py`

## 3. Données utilisées

Only P3 L1 partitions are used:

- `outputs/partitions/l1_binary/alpha_0.5/k3/client_i/train_scaled.npz`
- `outputs/partitions/l1_binary/alpha_0.5/k3/client_i/val_scaled.npz`

The global test holdout:

- `outputs/preprocessed/l1_binary/test_scaled.npz`

The test holdout is never loaded by clients. It is only loaded server-side after model/threshold selection.

## 4. Métriques

Round metrics:

- loss, accuracy, macro-F1, attack recall, FPR/FNR, TP/TN/FP/FN;
- round time and aggregation time;
- model size and communication bytes.

Client metrics:

- train/val sample counts;
- local train/val loss;
- local accuracy, macro-F1, attack recall, FPR;
- fit/eval time;
- upload/download bytes.

Final metrics:

- validation threshold sweep;
- global test metrics;
- P4 vs P5.2 comparison.

## 5. Logging

The run writes:

- `logs/flower_server.log`
- `logs/flower_clients.log`
- `logs/run_console.log`

Flower runtime logs are streamed to terminal by `run_simulation(verbose_logging=True)`. P5.2 also writes explicit server/client lifecycle logs to files for reproducibility.

## 6. Bandwidth

P5.2 uses deterministic tensor payload accounting:

- `download_bytes = model_size_bytes * active_clients`
- `upload_bytes = model_size_bytes * active_clients`
- `total_round_bytes = upload_bytes + download_bytes`
- `cumulative_bytes += total_round_bytes`

This mirrors P5.1, but the counts are collected from Flower strategy callbacks.

## 7. Commandes

Verify:

`python experiments/qi-fl-ids-iot-final/src/scripts/05_2_verify_flower_l1_setup.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_flower.yaml`

Smoke Flower:

`python experiments/qi-fl-ids-iot-final/src/scripts/05_2_run_flower_l1_smoke.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_flower.yaml --runtime legacy-local --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000`

Full Flower, documented only:

`python experiments/qi-fl-ids-iot-final/src/scripts/05_2_run_flower_l1_smoke.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_flower.yaml --runtime legacy-local --mode full --alpha 0.5 --clients 3 --rounds 30`

Future Flower CLI upgrade path:

`flwr run experiments/qi-fl-ids-iot-final --stream`

The CLI upgrade path may require aligning the environment with Flower app packaging used by newer Flower docs.

## 8. Critères d’acceptation

- Flower version detected.
- Flower app APIs available.
- P3 L1 partitions found.
- P4 metrics found.
- global test holdout found and not used by clients.
- server/client modules import.
- verify setup passes without training.
- tests pass.
- smoke one round passes if Flower simulation runtime is available.
- no full/grid launched automatically.
- no Docker/dashboard/QI/legacy experiment modifications.

## 9. P5.2 boundary

P5.2 stops at true Flower L1 FedAvg code-ready plus smoke validation. P6 is not started.
