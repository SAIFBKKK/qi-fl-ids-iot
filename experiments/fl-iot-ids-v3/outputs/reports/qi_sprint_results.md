# QI Optimization Sprint Results

Date: 2026-05-03

## Implemented

- `qifa` is selectable from the experiment registry.
- QIFA reuses the `ReportingFedAvg` tracking/reporting path.
- QIFA supports client-weighted aggregation, diversity adjustment, epsilon
  coefficients, optional perturbation, `lambda_qifa`, `delta_perturbation`,
  `sigma_noise`, `perturbation_frequency` and `random_seed`.
- QIFA is equivalent to FedAvg when `lambda_qifa=0` and perturbation is off.
- QGA feature selection exists as an isolated `src/qi` module.
- Runtime masking integrates selected features into FL clients without modifying
  raw CSV files or refitting scalers.

## Feature Subset

Smoke QGA on `normal_noniid` selected 16 of the real 28 v3 features.

Artifact:

```text
artifacts/qi_feature_selection/normal_noniid/selected_features.json
```

Selected features:

1. `flow_duration`
2. `Protocol Type`
3. `Duration`
4. `Rate`
5. `syn_flag_number`
6. `rst_flag_number`
7. `psh_flag_number`
8. `fin_count`
9. `rst_count`
10. `HTTPS`
11. `DNS`
12. `SSH`
13. `UDP`
14. `ARP`
15. `ICMP`
16. `Min`

## Validations Run

- `python -m pytest tests/test_qifa_strategy.py -q`
- `python -m pytest tests/test_qi_feature_selection.py -q`
- `python -m src.scripts.run_qi_feature_selection --scenario normal_noniid --config configs/qi/qga_feature_selection.yaml --smoke`
- `python -m pytest tests -q`
- `python -m src.scripts.run_experiment --experiment exp_v3_qifa_normal_classweights --dry-run`
- `python -m src.scripts.run_experiment --experiment exp_v3_qga_qifa_normal_classweights --dry-run`
- `python -m src.scripts.run_experiment --experiment exp_v3_qifa_normal_classweights --rounds 1`

## One-Round QIFA Validation

The one-round QIFA run completed on `normal_noniid` with 3 clients. These values
are validation-smoke results, not final benchmark conclusions.

- Distributed loss: `1.2169430758436461`
- Accuracy: `0.5710170805196131`
- Macro-F1: `0.4817664681837707`
- Benign recall: `0.7389534552590419`
- False positive rate: `0.2610465447409581`
- Rare class recall: `0.16933259388492114`
- Rare Macro-F1: `0.1622240159759835`
- QIFA lambda: `0.15`
- QIFA diversity norm: `14.665630349859683`
- QIFA perturbation norm: `0.0`
- QIFA effective clients: `3.0`
- Update size bytes: `536472.0`
- Train time sec: `189.19132209999952`

## Pending Full Runs

Full FL benchmarks still need to be run on the target machine:

- `python -m src.scripts.run_experiment --experiment exp_v3_fedavg_normal_classweights`
- `python -m src.scripts.run_experiment --experiment exp_v3_qifa_normal_classweights`
- `python -m src.scripts.run_experiment --experiment exp_v3_qga_fedavg_normal_classweights`
- `python -m src.scripts.run_experiment --experiment exp_v3_qga_qifa_normal_classweights`

No full-run Macro-F1, rare recall, FPR, convergence speed or latency values are
claimed in this report until those runs complete.
