# P5.1 - Runtime Audit and Logging

## 1. Scope

P5.1 does not redo P5 and does not invalidate the existing full run. It audits the current FedAvg L1 runtime path and adds detailed execution logging for future smoke/full runs.

## 2. In-process FedAvg vs real Flower

The current P5 implementation is an in-process FedAvg simulator: the server object loads all clients in the same Python process, sends model state dictionaries in memory, runs local PyTorch training sequentially, aggregates with FedAvg, and writes deterministic CSV/JSON artifacts.

This differs from a real Flower deployment where a Flower server starts, clients connect over the Flower runtime, the framework logs client availability, strategy callbacks, RPC-like exchanges, and process/network overhead. The current implementation is scientifically valid for FedAvg metrics and bandwidth accounting, but it is not a distributed systems demonstration.

## 3. Why the full run is faster than older v3/Flower runs

The run is expected to be faster because:

- no Flower server/client runtime is started;
- no network transport or Docker orchestration is used;
- all clients are local in one Python process;
- the L1 model is small: 12,098 parameters and about 48,392 bytes of tensor payload;
- local_epochs is 1;
- the task is binary L1 instead of the older 34-class historical context.

## 4. Full mode sample validation

The current code now enforces this policy:

- `smoke` may use `max_samples_per_client`;
- `full` and `grid` ignore `max_samples_per_client`;
- future full/grid manifests include `full_uses_all_client_samples=true` when loaded samples match the P3 client manifest.

For the existing `alpha=0.5, K=3` full run, `metrics_clients.csv` contains earlier smoke rows at the beginning, then the full rows. The full rows match the P3 manifest counts:

| Client | P3 train | Full train row | P3 val | Full val row |
| --- | ---: | ---: | ---: | ---: |
| client_1 | 203,744 | 203,744 | 27,634 | 27,634 |
| client_2 | 82,691 | 82,691 | 23,457 | 23,457 |
| client_3 | 154,565 | 154,565 | 43,409 | 43,409 |

Conclusion: the successful full run can be considered valid. P5.1 also adds this validation retrospectively to the existing `alpha=0.5, K=3` run manifest and run summary.

## 5. Bandwidth formula confirmation

P5 uses deterministic payload accounting, not real network counters:

`download_bytes = model_size_bytes * K`

`upload_bytes = model_size_bytes * K`

`total_round_bytes = upload_bytes + download_bytes`

`cumulative_bytes = previous_cumulative_bytes + total_round_bytes`

For `alpha=0.5, K=3`, the model payload was 48,392 bytes:

- upload per round: 145,176 bytes;
- download per round: 145,176 bytes;
- total per round: 290,352 bytes;
- 30 rounds cumulative: 8,710,560 bytes.

This matches the existing `bandwidth_rounds.csv`.

## 6. Runtime logging added

Future smoke/full runs now write:

- compact round logs in the terminal;
- Flower-like lifecycle messages;
- optional client-level logs controlled by `logging.verbose_clients`;
- mirrored logs in `outputs/fl_l1_fedavg/alpha_{alpha}/k{k}/logs/run_console.log`.

The existing full run was not relaunched. A retrospective `run_console.log` was reconstructed from its clean `run_summary.json` so the current full artifact set has the same observable round summary format.

Expected round format:

`[Round 03/30] loss=... val_loss=... macro_f1=... attack_recall=... FPR=... bytes=... cum=...`

## 7. Remaining recommendation

If the final demo needs visible distributed orchestration, add a real Flower runtime wrapper later. For scientific comparison P4 vs P5, the in-process FedAvg path is acceptable and easier to reproduce; for operational demonstration, Flower server/client logs would be more persuasive.

## 8. P5.1 conclusion

P5.1 improves observability and validates that full mode is not smoke sampling. No P5 full rerun, grid run, Docker, dashboard, QI module, or legacy experiment was modified by this audit.
