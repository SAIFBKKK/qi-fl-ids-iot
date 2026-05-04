# QI Sprint Validation

Date: 2026-05-04

## Scope

Validated only the sprint QI scope:

- QIFA aggregation
- QGA/QI feature selection
- benchmark configuration for FedAvg, QIFA, QGA+FedAvg and QGA+QIFA

No PowerSGD/compression or microservice implementation is included.

## QIFA Validation

QIFA now implements the requested normalized formula:

```text
w_avg = FedAvg(results)
epsilon_k = ||w_k - w_avg|| / (||w_avg|| + 1e-8)
w_global = sum_k normalized((n_k / N) * (1 + lambda_qifa * epsilon_k)) * w_k
```

Optional perturbation is:

```text
w_global += delta_perturbation * eta
eta ~ N(0, sigma_noise^2)
```

Unit coverage:

- output shape matches FedAvg
- works with 1, 2 and 3 clients
- `lambda_qifa=0` with perturbation off equals FedAvg
- deterministic seed reproduces perturbation
- perturbation disabled applies no noise

## QGA/QI Feature Selection Validation

Smoke mode produced a K=15 subset from the real 28-feature pipeline.

Artifact:

```text
artifacts/qi_feature_selection/normal_noniid/selected_features.json
```

Selected features:

1. `Protocol Type`
2. `Duration`
3. `syn_flag_number`
4. `rst_flag_number`
5. `psh_flag_number`
6. `ack_flag_number`
7. `ack_count`
8. `fin_count`
9. `urg_count`
10. `HTTPS`
11. `UDP`
12. `ARP`
13. `Tot sum`
14. `Min`
15. `Number`

Smoke selector result:

- Input features: `28`
- Selected features: `15`
- Mode: `smoke`
- Generations: `3`
- Population size: `4`
- Mini-MLP epochs: `1`
- Best validation Macro-F1: `0.01797385620915033`

This is a smoke validation only. It is not a final benchmark result.

## Commands Run

```powershell
python -m pytest tests/test_qifa_strategy.py -q
python -m pytest tests/test_qi_feature_selector.py -q
python -m src.scripts.run_qi_feature_selection --scenario normal_noniid --config configs/qi/qga_feature_selection.yaml --smoke
```

## Full Test Pass

Completed:

```powershell
python -m pytest tests -q
```

Result: `55 passed, 1 warning`.

## One-Round QIFA Run

Completed:

```powershell
python -m src.scripts.run_experiment --experiment exp_qi_qifa_28f --rounds 1
```

Runtime summary:

- Status: `success`
- Completed rounds: `1 / 1`
- Duration: `205.67` sec
- Feature count: `28`
- Classes: `34`
- Clients: `3`

Round 1 metrics:

- Distributed loss: `1.2524465202774278`
- Accuracy: `0.5900285278076987`
- Macro-F1: `0.5149508829847733`
- Recall macro: `0.5533836775496236`
- Benign recall: `0.7904741094249078`
- False positive rate: `0.20952589057509213`
- Rare class recall: `0.07689141387920316`
- Rare Macro-F1: `0.12792261316656184`
- Train loss last: `0.8517212847909574`
- Train time sec: `162.63059429999976`
- Update size bytes: `536472.0`
- `qifa/diversity_mean`: `0.6665478886859959`
- `qifa/perturbation_applied`: `0.0`
- `qifa/weight_norm_delta`: `0.02878683698780018`

This validates execution and logging of the QIFA formula path. It is not a
complete 10-round benchmark.

## Pending Full Benchmark

Still to run for final scientific comparison:

```powershell
python -m src.scripts.run_experiment --experiment exp_qi_baseline_fedavg_28f
python -m src.scripts.run_experiment --experiment exp_qi_qifa_28f
python -m src.scripts.run_experiment --experiment exp_qi_qga15_fedavg
python -m src.scripts.run_experiment --experiment exp_qi_qga15_qifa
```

The full FL benchmark can be long on Windows/Ray and CPU. No 10-round results
are claimed until these runs complete.
