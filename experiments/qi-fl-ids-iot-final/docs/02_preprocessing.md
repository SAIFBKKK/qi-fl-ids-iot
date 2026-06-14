# P2 — Preprocessing Report

## 1. Objectif
Préparer les datasets L1 binaire et L2 famille sans entraînement, sans Dirichlet et sans preprocessing hors périmètre.

## 2. Entrées utilisées
- Source Parquet : `data/balancing_v3_fixed300k_outputs/balancing_v3_fixed300k_balanced.parquet`
- Artefacts P1 : `experiments/qi-fl-ids-iot-final/outputs/artifacts`

## 3. Vérification P1
- P1 détectée : `True`
- Features confirmées : `28`

## 4. Vérification du statut de scaling
- Statut : `raw_unscaled`
- Justification : 12/28 features have large ranges, means or standard deviations; RobustScaler train-only is required.
- Scaling appliqué : `True`

## 5. Construction du dataset L1 binaire équilibré
- Total : `630000`
- Normal : `300000`
- Attack : `330000`
- Sampling attaques : `10000` par classe.

## 6. Split L1 train/val/test
- Counts : `{'train': 441000, 'val': 94500, 'test': 94500}`

## 7. Scaling L1 train-only
- Scaler : `experiments/qi-fl-ids-iot-final/outputs/artifacts/scalers/l1_binary_robust_scaler.pkl`
- NPZ : `{'train': 'experiments/qi-fl-ids-iot-final/outputs/preprocessed/l1_binary/train_scaled.npz', 'val': 'experiments/qi-fl-ids-iot-final/outputs/preprocessed/l1_binary/val_scaled.npz', 'test': 'experiments/qi-fl-ids-iot-final/outputs/preprocessed/l1_binary/test_scaled.npz'}`

## 8. Construction du dataset L2 family attack-only
- Total : `9101350`
- Familles : `['BruteForce', 'DDoS', 'DoS', 'Malware', 'Mirai', 'Recon', 'Spoofing', 'Web-based']`
- Sampling : `False`

## 9. Split L2 train/val/test
- Counts : `{'train': 6370944, 'val': 1365202, 'test': 1365204}`

## 10. Scaling L2 train-only
- Scaler : `experiments/qi-fl-ids-iot-final/outputs/artifacts/scalers/l2_family_robust_scaler.pkl`
- NPZ : `{'train': 'experiments/qi-fl-ids-iot-final/outputs/preprocessed/l2_family/train_scaled.npz', 'val': 'experiments/qi-fl-ids-iot-final/outputs/preprocessed/l2_family/val_scaled.npz', 'test': 'experiments/qi-fl-ids-iot-final/outputs/preprocessed/l2_family/test_scaled.npz'}`

## 11. Anti-leakage
- L1 : `{'anti_leakage_valid': True, 'anti_leakage_id': 'row_id', 'overlap_counts': {'train_val': 0, 'train_test': 0, 'val_test': 0}, 'split_counts': {'train': 441000, 'val': 94500, 'test': 94500}, 'total_assigned': 630000}`
- L2 : `{'anti_leakage_valid': True, 'anti_leakage_id': 'row_id', 'overlap_counts': {'train_val': 0, 'train_test': 0, 'val_test': 0}, 'split_counts': {'train': 6370944, 'val': 1365202, 'test': 1365204}, 'total_assigned': 9101350}`

## 12. Artefacts générés
- `experiments/qi-fl-ids-iot-final/outputs/preprocessed/l1_binary/manifest.json`
- `experiments/qi-fl-ids-iot-final/outputs/preprocessed/l1_binary/sampling_report.json`
- `experiments/qi-fl-ids-iot-final/outputs/preprocessed/l2_family/manifest.json`
- `experiments/qi-fl-ids-iot-final/outputs/preprocessed/l2_family/distribution_report.json`
- `experiments/qi-fl-ids-iot-final/outputs/preprocessed/l2_family/family_mapping.json`
- `experiments/qi-fl-ids-iot-final/outputs/artifacts/scalers/l1_binary_robust_scaler.pkl`
- `experiments/qi-fl-ids-iot-final/outputs/artifacts/scalers/l2_family_robust_scaler.pkl`

## 13. Figures générées
- `experiments/qi-fl-ids-iot-final/outputs/figures/preprocessing/01_scaling_status_check.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/preprocessing/02_l1_binary_distribution_before_after.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/preprocessing/03_l1_attack_sampling_per_class.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/preprocessing/04_l1_train_val_test_distribution.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/preprocessing/05_l2_family_distribution_full_attack_only.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/preprocessing/06_l2_train_val_test_distribution.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/preprocessing/07_feature_scaling_before_after_l1.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/preprocessing/08_feature_scaling_before_after_l2.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/preprocessing/09_preprocessing_pipeline_l1_l2.png`

## 14. Risques restants
- Aucun warning restant.

## 15. Critères d’acceptation

| critere | ok |
| --- | --- |
| p1_validated_detected | True |
| parquet_source_found | True |
| features_28_confirmed | True |
| scaling_status_check_executed | True |
| l1_total_630000 | True |
| l1_normal_300000 | True |
| l1_attack_330000 | True |
| l1_attack_sampling_10000_each | True |
| l1_split_generated | True |
| l1_anti_leakage_valid | True |
| l1_npz_generated | True |
| l2_attack_only_no_sampling | True |
| l2_excludes_benign | True |
| l2_eight_families | True |
| l2_split_generated | True |
| l2_anti_leakage_valid | True |
| l2_npz_generated | True |
| manifests_generated | True |
| reports_generated | True |
| figures_9_generated | True |

## 16. Conclusion P2

P2 est validée.
