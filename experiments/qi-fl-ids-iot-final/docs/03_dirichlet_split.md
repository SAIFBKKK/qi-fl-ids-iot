# P3 — Dirichlet Split Report

## 1. Objectif
Créer des partitions fédérées non-IID pour L1 binaire et L2 family attack-only, sans entraîner de modèle.

## 2. Entrées utilisées
- L1 train/val : `experiments/qi-fl-ids-iot-final/outputs/preprocessed/l1_binary/train_scaled.npz`, `experiments/qi-fl-ids-iot-final/outputs/preprocessed/l1_binary/val_scaled.npz`
- L2 train/val : `experiments/qi-fl-ids-iot-final/outputs/preprocessed/l2_family/train_scaled.npz`, `experiments/qi-fl-ids-iot-final/outputs/preprocessed/l2_family/val_scaled.npz`

## 3. Rappel méthodologique Dirichlet
Chaque classe est mélangée puis distribuée entre clients selon un tirage Dirichlet. Plus alpha est faible, plus les clients sont hétérogènes.

## 4. Choix des alpha
- `0.1` : non-IID extrême / stress-test.
- `0.5` : scénario principal réaliste.
- `5.0` : quasi-IID / référence stable.

## 5. Choix des clients K
Les scénarios couvrent `K ∈ {3, 4, 5}`.

## 6. Règle deployment : global test holdout
Train et val sont partitionnés pour FL. Le test global n’est pas partitionné et reste réservé à l’évaluation finale du modèle global, à la validation offline, aux tests microservices et à la simulation dashboard/inference.

## 7. Partitionnement L1 binaire
- Scénarios générés : `9`.
- Stockage : NPZ matérialisés par client pour train/val.

## 8. Partitionnement L2 family attack-only
- Scénarios générés : `9`.
- Stockage : index_only via `train_row_ids.npy` et `val_row_ids.npy`.

## 9. Anti-leakage
- Scénarios valides : `18/18`.

## 10. Distributions par client
Chaque scénario contient `distribution_report.json` et `client_distribution.csv`.

## 11. Figures générées
- `experiments/qi-fl-ids-iot-final/outputs/figures/dirichlet_split/01_l1_samples_per_client_alpha_k.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/dirichlet_split/02_l1_binary_heatmaps.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/dirichlet_split/03_l1_alpha_comparison.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/dirichlet_split/04_l2_samples_per_client_alpha_k.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/dirichlet_split/05_l2_family_heatmaps.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/dirichlet_split/06_l2_alpha_comparison.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/dirichlet_split/07_global_test_holdout_explanation.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/dirichlet_split/08_dirichlet_pipeline_l1_l2.png`

## 12. Artefacts générés
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l1_binary/alpha_0.1/k3`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l2_family/alpha_0.1/k3`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l1_binary/alpha_0.1/k4`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l2_family/alpha_0.1/k4`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l1_binary/alpha_0.1/k5`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l2_family/alpha_0.1/k5`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l1_binary/alpha_0.5/k3`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l2_family/alpha_0.5/k3`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l1_binary/alpha_0.5/k4`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l2_family/alpha_0.5/k4`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l1_binary/alpha_0.5/k5`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l2_family/alpha_0.5/k5`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l1_binary/alpha_5.0/k3`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l2_family/alpha_5.0/k3`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l1_binary/alpha_5.0/k4`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l2_family/alpha_5.0/k4`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l1_binary/alpha_5.0/k5`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l2_family/alpha_5.0/k5`

## 13. Risques restants
- Aucun warning restant.

## 14. Critères d’acceptation

| critere | ok |
| --- | --- |
| p2_validated_detected | True |
| l1_9_scenarios_generated | True |
| l2_9_scenarios_generated | True |
| alphas_processed | True |
| client_counts_processed | True |
| l1_train_val_partitioned | True |
| l1_test_global_not_partitioned | True |
| l1_global_test_references | True |
| l1_client_npz_generated | True |
| l2_train_val_index_only | True |
| l2_test_global_not_partitioned | True |
| l2_global_test_references | True |
| l2_row_id_indexes_generated | True |
| no_client_empty | True |
| anti_leakage_valid | True |
| manifests_generated | True |
| distribution_reports_generated | True |
| client_distribution_csv_generated | True |
| figures_p3_generated | True |

## 15. Conclusion P3

P3 est validée.
