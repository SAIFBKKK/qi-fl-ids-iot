# P8.1.5 — QGA Engineering Calibration & Robustness Plan

## 1. Objectif

P8.1.5 introduit une étape de calibration avant de figer définitivement le masque QGA L1. L'objectif est de comparer plusieurs profils d'optimisation, plusieurs seeds et plusieurs validations Flower courtes afin d'éviter de valider P8 sur un seul masque et un seul scénario.

## 2. Principe général

La sélection du masque reste un processus QGA standalone fondé uniquement sur L1 train/validation. Le global test holdout n'est pas utilisé pour sélectionner le masque.

Les validations FL de robustesse utilisent ensuite le vrai runtime Flower FedAvg + QGA L1, en mode court, avec logs, run_id, artefacts et `test_sent_to_clients=false`.

## 3. Profils QGA

Les profils configurés sont :

| Profil | Macro-F1 | Attack recall | Feature penalty | FPR penalty | Bounds |
|---|---:|---:|---:|---:|---|
| balanced_current | 0.60 | 0.30 | 0.10 | 0.00 | 8-24 |
| conservative | 0.70 | 0.20 | 0.10 | 0.00 | 12-24 |
| fpr_aware | 0.55 | 0.25 | 0.10 | 0.10 | 10-24 |
| compression | 0.50 | 0.20 | 0.30 | 0.00 | 8-18 |

Seeds : `42`, `123`, `2026`.

## 4. Sweep QGA

Le sweep exécute `profile x seed`, sauvegarde chaque run et produit :

- `outputs/reports/p8_qga_profile_sweep_summary.csv`
- `outputs/reports/p8_qga_profile_sweep_summary.json`
- `outputs/reports/p8_qga_mask_stability.json`

Commande complète :

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/08_1_5_run_qga_profile_sweep.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml
```

## 5. Validation Flower courte

La validation Flower courte prend les meilleurs masques candidats et exécute FedAvg + QGA avec un vrai runtime Flower.

Scénarios initiaux :

- alpha=0.1, K=3
- alpha=0.5, K=3
- alpha=5.0, K=3

Scénarios complémentaires pour le meilleur masque :

- alpha=0.5, K=4
- alpha=0.5, K=5

Commande :

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/08_1_5_run_qga_flower_short_validation.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml --top-n 3 --rounds 5 --max-samples-per-client 1000 --address 127.0.0.1:8083
```

## 6. Sélection du masque final

Le score d'ingénierie est :

```text
0.40 * mean_macro_f1
+ 0.25 * mean_attack_recall
- 0.20 * mean_fpr
- 0.10 * std_macro_f1
- 0.05 * feature_ratio
```

La sélection ne doit être lancée qu'après un sweep et des validations Flower suffisamment complets.

Commande :

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/08_1_5_select_best_qga_mask.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml
```

## 7. Rapport et figures

Le rapport de calibration est généré par :

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/08_1_5_build_qga_calibration_report.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml
```

Figures attendues :

- `qga_profiles_macro_f1.png`
- `qga_profiles_features_count.png`
- `qga_profiles_fpr.png`
- `qga_profile_fitness_boxplot.png`
- `qga_short_flower_macro_f1_by_scenario.png`
- `qga_short_flower_fpr_by_scenario.png`
- `qga_mask_stability_heatmap.png`
- `qga_engineering_score_ranking.png`

## 8. Critères d'acceptation

- Les profils QGA sont configurés.
- Le sweep QGA produit des résumés consolidés.
- Les validations FL courtes utilisent le vrai runtime Flower.
- Le global test n'est pas utilisé pour sélectionner le masque.
- Les figures et le rapport P8.1.5 sont générés.
- Aucun full 30 rounds, QIFA, FedTN, Docker, dashboard ou P8-b n'est lancé par Codex.

## 9. Statut

Plan prêt. Un mini-run de démonstration peut valider la chaîne, mais la sélection finale du masque doit attendre le sweep complet et les validations Flower 5 rounds décidées par l'utilisateur.
