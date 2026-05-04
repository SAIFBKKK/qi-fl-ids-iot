# QIFA Strategy

QIFA means Quantum-Inspired Federated Averaging. In this repository it is a
controlled FL aggregation strategy for `experiments/fl-iot-ids-v3`, not a
quantum hardware implementation.

## Scope

QIFA is implemented only in:

```text
experiments/fl-iot-ids-v3/src/fl/qifa_strategy.py
```

No competing `qifa.py` module is used. The strategy subclasses
`ReportingFedAvg`, so FedAvg/FedProx/SCAFFOLD paths keep their existing
tracking, reporting and checkpoint behavior.

## Formula

QIFA starts from standard client-weighted FedAvg:

```text
w_avg = sum_k (n_k / N) * w_k
```

For each client:

```text
epsilon_k = ||w_k - w_avg|| / (||w_avg|| + 1e-8)
```

The effective client weight is:

```text
a_k = (n_k / N) * (1 + lambda_qifa * epsilon_k)
```

The effective weights are normalized before aggregation:

```text
a_k = a_k / sum_j a_j
w_global = sum_k a_k * w_k
```

This normalization prevents numerical weight explosion when client diversity is
large.

## Optional Perturbation

When enabled and the round matches `perturbation_frequency`:

```text
w_global = w_global + delta_perturbation * eta
eta ~ N(0, sigma_noise^2)
```

The perturbation RNG is seeded with `random_seed + server_round`, making smoke
tests reproducible.

## Config

QIFA parameters live under `strategy.qifa`:

```yaml
qifa:
  lambda_qifa: 0.15
  perturbation_enabled: false
  delta_perturbation: 0.01
  sigma_noise: 0.001
  perturbation_frequency: 2
  random_seed: 42
```

The canonical QI sprint config is:

```text
experiments/fl-iot-ids-v3/configs/fl/qifa_formula.yaml
```

## FedAvg Equivalence

This control case is mandatory and tested:

```yaml
qifa:
  lambda_qifa: 0.0
  perturbation_enabled: false
```

With that configuration, QIFA returns the same tensors as weighted FedAvg.

## Metrics

QIFA emits:

- `qifa/diversity_mean`
- `qifa/perturbation_applied`
- `qifa/weight_norm_delta`
- `qifa_lambda`
- `qifa_diversity_norm`
- `qifa_perturbation_norm`
- `qifa_effective_clients`

The first three are the sprint-facing metrics. The underscore metrics are kept
for compatibility with existing round report conventions.
