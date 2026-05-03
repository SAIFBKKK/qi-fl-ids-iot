# QIFA Strategy

QIFA means Quantum-Inspired Federated Averaging. In this repository it is a
controlled FL strategy extension for `experiments/fl-iot-ids-v3`, not a quantum
hardware implementation.

## Integration

- Strategy module: `experiments/fl-iot-ids-v3/src/fl/qifa_strategy.py`
- Server selection: `experiments/fl-iot-ids-v3/src/fl/server_app.py`
- Config: `experiments/fl-iot-ids-v3/configs/fl/qifa.yaml`
- Registry key: `fl_strategy: qifa`, `fl_config: qifa`

QIFA subclasses `ReportingFedAvg`, so the existing metric aggregation,
round tracking, MLflow logging and best-checkpoint path remain shared with the
validated FedAvg path.

## Aggregation

The aggregation starts from client-weighted FedAvg:

```text
w_avg = sum_i (n_i / N) * w_i
```

Then QIFA computes a diversity vector from each client update:

```text
d_i = w_i - w_avg
```

The final update is:

```text
w_qifa = w_avg + lambda_qifa * sum_i alpha_i * epsilon_i * diversity_i * d_i + perturbation
```

where `alpha_i = n_i / N`. Perturbation is optional and deterministic when a
fixed `random_seed` is used.

## FedAvg Equivalence

The mandatory control case is implemented and tested:

```yaml
qifa:
  lambda_qifa: 0.0
  perturbation_enabled: false
```

With this config, QIFA returns the same parameter tensors as weighted FedAvg.

## Round Metrics

QIFA logs these fit-round metrics:

- `qifa_lambda`
- `qifa_diversity_norm`
- `qifa_perturbation_norm`
- `qifa_effective_clients`

MLflow aliases are emitted under the `qifa/` namespace.
