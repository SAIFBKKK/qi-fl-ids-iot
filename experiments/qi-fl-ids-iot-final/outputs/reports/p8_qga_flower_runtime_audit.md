# P8 QGA FL Runtime Audit

## 1. Files inspected

| File | Finding |
|---|---|
| `src/scripts/08_run_qga_fedavg_l1.py` | Calls `qga.fedavg_adapter.run_qga_fedavg_l1`; no Flower server/client is started. |
| `src/scripts/08_run_qga_heterofl_l1.py` | Calls `qga.heterofl_adapter.run_qga_heterofl_l1`; no Flower server/client is started. |
| `src/qga/fedavg_adapter.py` | In-process FedAvg loop with local model copies and direct Python aggregation. |
| `src/qga/heterofl_adapter.py` | In-process HeteroFL-style loop using P7 slicing and aggregation. |
| `src/fl_l1_flower/legacy_server.py` | True Flower server path using `fl.server.start_server`. |
| `src/fl_l1_flower/legacy_client.py` | True Flower client path using `fl.client.start_client`. |
| `src/fl_l1_flower/strategy.py` | Rich Flower run contract: run_id, logs, metrics, bandwidth, checkpoints, run_summary. |
| `experiments/fl-iot-ids-v3/src/fl/` | Historical Flower inspiration only; final runtime should reuse P5.2.1/P5.2.2 conventions. |

## 2. Runtime classification

- QGA standalone is valid and unchanged.
- `08_run_qga_fedavg_l1.py` is **in-process**, not a true Flower runtime.
- `08_run_qga_heterofl_l1.py` is **in-process**, not a true Flower runtime.

## 3. Required correction

Starting from P8, final FL trainings must use true Flower. Therefore, P8 FedAvg + QGA needs a separate real Flower path with:

- real Flower server;
- real Flower clients;
- run-specific `run_id`;
- separate server/client logs;
- rich `run_summary.json`;
- `true_flower_runtime=true`;
- `test_sent_to_clients=false`.

## 4. HeteroFL + QGA status

HeteroFL + QGA remains in-process experimental for now. A true Flower HeteroFL runtime is more complex because clients have different model widths and Flower FedAvg assumes homogeneous parameter shapes. It must not be reported as the final true Flower baseline.

## 5. Recommendation

Keep the existing P8 in-process scripts as experimental helpers, add true Flower FedAvg + QGA L1 server/client/smoke scripts, and document that final P8 FedAvg + QGA must use the new Flower path.
