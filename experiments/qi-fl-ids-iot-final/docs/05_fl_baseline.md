# P5 — Federated L1 FedAvg Baseline

## 1. Objectif
Créer la baseline Federated Learning L1 avec FedAvg pour comparaison avec P4 centralisé.

## 2. Réutilisation contrôlée depuis v3
Voir `outputs/reports/p5_reuse_notes.md`. Les patterns FedAvg, logging et communication tracking ont été repris conceptuellement, puis réimplémentés pour les artefacts finaux.

## 3. Données utilisées
P5 utilise les partitions L1 P3 pour train/val client. Le global test holdout P2 reste intact.

## 4. Rappel P4 centralisé
P4 sert de référence centralisée L1 et fournit `metrics_test.json`, `threshold.json` et `model_config.json`.

## 5. Principe FedAvg
`w_global = sum(n_k / n_total * w_k)` avec pondération par nombre d'exemples client.

## 6. Scénarios Dirichlet
La grille prévue est `alpha ∈ {0.1, 0.5, 5.0}` et `K ∈ {3, 4, 5}`.

## 7. Configuration FL
FedAvg, 30 rounds, local_epochs=1, client_fraction=1.0, batch_size=512.

## 8. Architecture du modèle
`28 -> 128 -> 64 -> 2`, alignée avec P4.

## 9. Monitoring et communication cost
Le code trace round metrics, client metrics, bandwidth, model size, aggregation time et latency.

## 10. Métriques par round
`metrics_rounds.csv` est généré après un run smoke/full/grid.

## 11. Métriques par client
`metrics_clients.csv` est généré après un run smoke/full/grid.

## 12. Threshold tuning
Le threshold est tuné sur validation fédérée uniquement.

## 13. Évaluation global test holdout
Le test global n'est évalué qu'après sélection du modèle et du threshold.

## 14. Comparaison P4 vs P5
`comparison_with_p4.json` est prêt après run et compare accuracy, macro-F1, attack recall et FPR.

## 15. Impact de alpha
À analyser après full/grid run utilisateur.

## 16. Impact de K
À analyser après full/grid run utilisateur.

## 17. Artefacts générés
- Code P5 prêt.
- Verify summary : `outputs/reports/fl_l1_fedavg_verify_summary.json`.
- Reuse notes : `outputs/reports/p5_reuse_notes.md`.

## 18. Figures générées
Les figures FL sont générées après smoke/full/grid run.

## 19. Commandes d’exécution
- Verify : `python experiments/qi-fl-ids-iot-final/src/scripts/05_verify_fl_l1_setup.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_fedavg.yaml`
- Smoke : `python experiments/qi-fl-ids-iot-final/src/scripts/05_train_fl_l1_fedavg.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_fedavg.yaml --mode smoke --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000`
- Full : `python experiments/qi-fl-ids-iot-final/src/scripts/05_train_fl_l1_fedavg.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_fedavg.yaml --mode full --alpha 0.5 --clients 3 --rounds 30`
- Grid : `python experiments/qi-fl-ids-iot-final/src/scripts/05_train_fl_l1_fedavg.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_fedavg.yaml --mode grid`

## 20. Critères d’acceptation

| critere | ok |
| --- | --- |
| config_loads | True |
| p3_l1_partitions_exist | True |
| default_scenario_loads | True |
| global_test_holdout_exists | True |
| global_test_not_partitioned | True |
| p4_metrics_exist | True |
| p4_threshold_exist | True |
| p4_model_config_exist | True |
| model_architecture_matches_p4 | True |
| fedavg_aggregation_ready | True |
| bandwidth_tracking_ready | True |
| logging_configured | True |
| full_mode_does_not_use_smoke_sampling | True |
| full_uses_all_client_samples | True |
| verify_runs_without_training | True |

## 21. Conclusion P5

P5 est code-ready. Aucun entraînement complet ni grid n'a été lancé par Codex; verify/smoke/full/grid sont disponibles pour exécution utilisateur.
