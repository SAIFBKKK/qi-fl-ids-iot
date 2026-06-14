# P4 — Centralized L1 Binary Baseline

## 1. Objectif
Entraîner une baseline centralisée L1 binaire normal vs attack, référence directe pour P5 FL L1.

## 2. Données utilisées
- Train : `441000` rows.
- Val : `94500` rows.
- Test holdout : `94500` rows.

## 3. Baseline historique Kaggle 34 classes
- Architecture : `28 -> 128 -> 64 -> 34`.
- Best val macro-F1 : `0.8110840586599894`.
- Test macro-F1 : `0.8109757023708111`.

## 4. Pourquoi une nouvelle baseline L1 est nécessaire
La baseline Kaggle est multiclasses L3/34 classes. P4 entraîne un modèle binaire dédié à la production L1 et à la comparaison future avec FL.

## 5. Architecture du modèle L1
- `28 -> 128 -> 64 -> 2`.
- Paramètres : `12098`.

## 6. Configuration d'entraînement
- Optimizer : `adam`.
- Batch size : `512`.
- Device : `cuda`.

## 7. Stratégie de validation
Le meilleur checkpoint est sélectionné uniquement sur validation macro-F1. Le test global n’est pas utilisé pendant l'entraînement.

## 8. Threshold tuning
- Primary threshold : `0.5`.
- Le threshold est choisi uniquement sur validation.

## 9. Résultats validation
- Best epoch : `30`.
- Val macro-F1 : `0.961596475833504`.
- Val attack-F1 : `0.9630407734513635`.

## 10. Résultats test global holdout
- Accuracy : `0.9611957671957672`.
- Macro-F1 : `0.961143284995805`.
- Attack recall : `0.9525858585858585`.

## 11. Matrice de confusion
- TP `47153`, TN `43680`, FP `1320`, FN `2347`.

## 12. Analyse TP / TN / FP / FN
Les TP/TN/FP/FN sont calculés avec attack comme classe positive.

## 13. FPR et FNR
- FPR : `0.029333333333333333`.
- FNR : `0.047414141414141416`.

## 14. Latence et taille modèle
- Latence : `0.017730554497262425` ms/sample.
- Taille modèle : `51622` bytes.

## 15. Artefacts générés
- `experiments/qi-fl-ids-iot-final/outputs/centralized_l1/checkpoints/best_model.pth`
- `experiments/qi-fl-ids-iot-final/outputs/centralized_l1/checkpoints/last_model.pth`
- `experiments/qi-fl-ids-iot-final/outputs/centralized_l1/artifacts/model_config.json`
- `experiments/qi-fl-ids-iot-final/outputs/centralized_l1/artifacts/threshold.json`
- `experiments/qi-fl-ids-iot-final/outputs/centralized_l1/artifacts/metrics_val.json`
- `experiments/qi-fl-ids-iot-final/outputs/centralized_l1/artifacts/metrics_test.json`
- `experiments/qi-fl-ids-iot-final/outputs/centralized_l1/artifacts/classification_report.json`
- `experiments/qi-fl-ids-iot-final/outputs/centralized_l1/artifacts/confusion_matrix.csv`
- `experiments/qi-fl-ids-iot-final/outputs/centralized_l1/artifacts/threshold_sweep.csv`
- `experiments/qi-fl-ids-iot-final/outputs/centralized_l1/artifacts/training_history.csv`
- `experiments/qi-fl-ids-iot-final/outputs/centralized_l1/logs/training_history.csv`
- `experiments/qi-fl-ids-iot-final/outputs/centralized_l1/artifacts/historical_kaggle_34class_baseline.json`

## 16. Figures générées
- `experiments/qi-fl-ids-iot-final/outputs/figures/training/centralized_l1_loss_curve.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/training/centralized_l1_accuracy_curve.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/training/centralized_l1_f1_curve.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/evaluation/centralized_l1_confusion_matrix.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/evaluation/centralized_l1_threshold_sweep.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/evaluation/centralized_l1_tp_tn_fp_fn.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/evaluation/centralized_l1_roc_curve.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/evaluation/centralized_l1_pr_curve.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/models/centralized_l1_architecture.png`

## 17. Comparaison attendue avec P5 FL
P5 devra comparer FedAvg L1 au modèle centralisé P4 sur le même test global holdout.

## 18. Risques restants
- Aucun warning restant.

## 19. Critères d’acceptation

| critere | ok |
| --- | --- |
| p2_validated_detected | True |
| train_val_test_l1_loaded | True |
| input_dim_28_confirmed | True |
| output_dim_2_confirmed | True |
| model_trained_without_error | True |
| best_model_saved | True |
| last_model_saved | True |
| training_history_generated | True |
| best_checkpoint_validation_only | True |
| threshold_validation_only | True |
| test_evaluated_after_selection | True |
| metrics_val_generated | True |
| metrics_test_generated | True |
| threshold_json_generated | True |
| threshold_sweep_generated | True |
| confusion_matrix_generated | True |
| tp_tn_fp_fn_calculated | True |
| core_metrics_calculated | True |
| fpr_fnr_calculated | True |
| latency_calculated | True |
| model_size_calculated | True |
| num_parameters_calculated | True |
| historical_kaggle_artifact_generated | True |
| docs_generated | True |
| figures_generated | True |
| dirichlet_partitions_not_used | True |

## 20. Conclusion P4

P4 est validée.
