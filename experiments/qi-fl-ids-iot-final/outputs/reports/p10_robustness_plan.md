# P10 Robustness Plan

## Objective

Evaluate L1 binary IDS robustness under controlled perturbations and poisoning for:

- FedAvg
- FedAvg + QGA
- QIFA
- QIFA + QGA

## Simulated Attacks

- `clean`: no poisoning, reference scenario.
- `label_flip`: invert a proportion of local training labels on compromised clients.
- `attack_to_normal`: hide attacks by converting a proportion of attack labels to normal on compromised clients.
- `feature_noise`: inject bounded Gaussian noise into local training features.

## Scientific Rules

- Poisoning is applied only to local client train arrays.
- Local validation is clean by default.
- Global test holdout is never modified and never sent to clients.
- P3 partition files are never overwritten.
- Docker, dashboard, FedTN/MPS, and P11 are outside P10.

## Scenarios

Default scenario:

- alpha = 0.5
- K = 3
- rounds = 30
- attack_type = `label_flip`
- poison_rate = 0.2
- poisoned_clients = 1

Code-ready smoke uses one round and at most 1000 samples per client.

## Metrics

P10 tracks macro-F1, attack recall, FPR, FNR, accuracy, confusion matrix, client metrics, round metrics, aggregation weights, and a defensive robustness score:

`0.5 * macro_f1 + 0.3 * attack_recall - 0.2 * FPR`

## Outputs

Run outputs are written under:

`outputs/robustness_l1/{method}/alpha_{alpha}/k{k}/{attack_type}/rate_{poison_rate}/clients_{poisoned_clients}/runs/{run_id}/`

Reports and figures are written under:

- `outputs/reports/p10_robustness_*`
- `outputs/figures/robustness_l1/`

## Full Runtime Path

The lightweight runner validates poisoning logic. Important final QIFA and QIFA+QGA runs should be launched manually through the existing Flower runtimes, with the same poisoning manifest contract.
