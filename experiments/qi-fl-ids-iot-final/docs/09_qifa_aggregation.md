# P9 — QIFA Aggregation

## 1. Objectif
Tester une agrégation Quantum-Inspired pour L1 en vrai runtime Flower.

## 2. Formulation
Score client :

`s_i = a * MacroF1_i + b * RecallAttack_i - c * FPR_i - d * Loss_i - e * Drift_i`

Transformation :

`theta_i = normalize(s_i)`

`amplitude_i = sin(theta_i)`

`probability_i = amplitude_i^2 / sum_j amplitude_j^2`

`omega_i = (1 - gamma) * fedavg_weight_i + gamma * probability_i`

## 3. Runtime
- vrai serveur Flower
- vrais clients Flower
- `test_sent_to_clients=false`
- évaluation finale uniquement côté serveur

## 4. Variantes
- `performance`
- `loss_aware`
- `drift_aware`
- `hybrid`

## 5. Option QGA
- QIFA seul : 28 features
- QIFA + QGA : masque calibré P8 `conservative_seed_42`, 12 features

## 6. Artefacts
- `qifa_scores.csv`
- `qifa_probabilities.csv`
- `qifa_amplitudes.csv`
- `qifa_entropy.csv`
- `aggregation_weights.csv`
- `metrics_rounds.csv`
- `metrics_test.json`

## 7. Conclusion
P9 est préparé comme baseline Flower L1 expérimentale pour comparer FedAvg, FedAvg+QGA, QIFA, et QIFA+QGA sans toucher au dashboard ni aux pipelines de production.
