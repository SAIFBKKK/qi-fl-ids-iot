# P5.2.2 - Flower Output Contract Audit

## 1. Scope

Audited files:

- `experiments/qi-fl-ids-iot-final/src/fl_l1_flower/report_builder.py`
- `experiments/qi-fl-ids-iot-final/src/fl_l1_flower/strategy.py`
- `experiments/qi-fl-ids-iot-final/src/fl_l1_flower/metrics.py`
- `experiments/qi-fl-ids-iot-final/src/fl_l1_flower/communication.py`
- `experiments/qi-fl-ids-iot-final/src/fl_l1_flower/runtime.py`
- `experiments/qi-fl-ids-iot-final/src/fl_l1_flower/legacy_server.py`
- `experiments/qi-fl-ids-iot-final/src/fl_l1_flower/legacy_client.py`
- `experiments/qi-fl-ids-iot-final/src/scripts/05_2_verify_flower_l1_setup.py`
- `experiments/qi-fl-ids-iot-final/src/scripts/05_2_run_flower_l1_smoke.py`
- `experiments/qi-fl-ids-iot-final/src/scripts/05_2_start_flower_server.py`
- `experiments/qi-fl-ids-iot-final/src/scripts/05_2_start_flower_client.py`
- existing outputs under `experiments/qi-fl-ids-iot-final/outputs/fl_l1_flower/`

## 2. What Already Exists

The P5.2/P5.2.1 runtime already produces a real Flower execution path with:

- run-specific directories under `outputs/fl_l1_flower/alpha_0.5/k3/runs/{run_id}/`;
- per-run logs: `flower_server.log`, `flower_clients.log`, `run_console.log`;
- per-run CSVs: `metrics_rounds.csv`, `metrics_clients.csv`, `bandwidth_rounds.csv`, `aggregation_weights.csv`;
- validation/test JSON metrics;
- threshold JSON and threshold sweep CSV;
- confusion matrix CSV;
- comparison with P4 JSON;
- best and last checkpoints;
- root-level `manifest.json` per run with `test_sent_to_clients=false`;
- scenario-level `latest_run.json`.

The verify path already checks Flower availability, P3 partitions, P4 metrics, scenario clients, and global test holdout protection.

## 3. What Is Missing

The current verify summary is technically useful but too small for final reporting. It lacks:

- stable `artifacts_expected` and `figures_expected` lists;
- explicit `criteria` flags;
- explicit `errors` list;
- a canonical filename matching P5.2: `fl_l1_flower_verify_summary.json`.

The current run summary is too flat and lacks:

- `accepted`;
- nested `scenario`, `dataset`, `model`, `training`, `validation`, `test`;
- P4 comparison with accuracy gaps;
- `artifacts` list containing only existing files;
- `figures` list containing only existing figures;
- `criteria` flags;
- `warnings` and `errors`;
- `classification_report.json`;
- `model_config.json`;
- `run_manifest.json` inside `artifacts/`;
- scenario-level `latest_run_summary.json`;
- generated Flower figures under `outputs/figures/fl_l1_flower/alpha_{alpha}/k{k}/{run_id}/`.

## 4. Inconsistencies

- Some legacy smoke artifacts still exist directly under `outputs/fl_l1_flower/alpha_0.5/k3/`, while P5.2.1 moved new executions under `runs/{run_id}`. New P5.2.2 outputs should use the run-specific layout only.
- The old verify output name `flower_l1_verify_summary.json` does not match the requested P5.2 name `fl_l1_flower_verify_summary.json`. P5.2.2 should write the new file and keep the old file as a compatibility alias.
- Existing `selection_split` in threshold payload may say `validation`; P5.2 reports should normalize this to `server_validation`.
- Existing smoke metrics can be generated from sampled clients/test data and are not scientifically meaningful. The run summary must mark `scientific_significance=low_for_smoke`.

## 5. Reference Output Structure

P5.2.2 makes two contracts authoritative:

1. Verify summary:
   - readiness-only;
   - no training;
   - contains checks, expected artifacts/figures, criteria, warnings, and errors.

2. Run summary:
   - one JSON per smoke/full execution;
   - nested P4-style sections;
   - all artifact and figure paths must point to files that exist after the run;
   - scenario-level `latest_run_summary.json` mirrors the latest run for reporting.

## 6. Recommendation

Implement a small schema helper and plotting helper inside `src/fl_l1_flower/`, then call them from `strategy.finalize()`. This keeps P5/P5.1/P5.2/P5.2.1 behavior intact while enriching Flower-specific outputs.
