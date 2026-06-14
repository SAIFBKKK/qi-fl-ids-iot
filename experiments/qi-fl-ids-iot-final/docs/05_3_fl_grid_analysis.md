# P5.3 — FedAvg L1 Grid Study

## 1. Objectif
Étudier l'influence de `alpha` et `K` sur les performances de FedAvg L1.

## 2. Effet de alpha
Les résultats de la grid FedAvg montrent que le paramètre Dirichlet `alpha` influence fortement la stabilité de l'apprentissage fédéré. Lorsque `alpha` est faible, les distributions locales deviennent fortement hétérogènes, ce qui accentue le client drift et dégrade la convergence globale.

Cette fragilité est particulièrement visible pour `alpha=0.1` et `K=5`, où le modèle obtient un Macro-F1 de `0.8085` et un FPR de `0.3256`. Ce scénario représente donc le stress-test non-IID le plus difficile pour FedAvg.

À l'inverse, lorsque `alpha` augmente, les distributions clients deviennent plus proches de la distribution globale. Le client drift diminue, FedAvg converge plus facilement, et les performances deviennent plus stables. Le régime `alpha=5.0` présente les meilleurs résultats globaux et la meilleure stabilité.

## 3. Effet de K
L'effet du nombre de clients `K` dépend du niveau d'hétérogénéité. Quand `K` augmente, plus de clients participent et le coût de communication augmente. Les données sont aussi plus fragmentées, ce qui peut accroître la variance entre clients.

Pour `alpha=0.5`, augmenter `K` améliore les performances, probablement grâce à une meilleure diversité des données agrégées. En revanche, pour `alpha=0.1`, l'augmentation du nombre de clients amplifie la fragmentation des données et entraîne une forte instabilité, notamment pour `K=5`.

## 4. Communication / bandwidth
Formule utilisée : `C_round = 2 × K × model_size` avec `model_size = 48,392 bytes`.

- K=3 -> 290,352 bytes/round -> 8,710,560 bytes pour 30 rounds
- K=4 -> 387,136 bytes/round -> 11,614,080 bytes pour 30 rounds
- K=5 -> 483,920 bytes/round -> 14,517,600 bytes pour 30 rounds

## 5. Tableau comparatif
Tableau global : `experiments/qi-fl-ids-iot-final/outputs/reports/p5_grid_comparative_table.md`.

## 6. Discussion
Le meilleur scénario global est `alpha=5.0, K=3`, avec un Macro-F1 de `0.9545`, un attack recall de `0.9481` et un FPR de `0.0384`.

Ce scénario offre le meilleur compromis performance/coût : il obtient la meilleure performance globale tout en gardant le coût de communication minimal parmi les valeurs de `K`, car `K=3` implique le plus faible volume d'échange par round.

Le scénario `alpha=0.5, K=5` reste toutefois intéressant comme scénario réaliste non-IID performant. Il pourra servir de point de comparaison utile pour les prochaines phases, notamment Multi-tier, QIFA et les méthodes Quantum-Inspired, car il combine une hétérogénéité modérée avec un plus grand nombre de clients.

Figures générées :
- `experiments/qi-fl-ids-iot-final/outputs/figures/fl_l1_grid/heatmap_macro_f1_alpha_k.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/fl_l1_grid/heatmap_attack_recall_alpha_k.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/fl_l1_grid/heatmap_fpr_alpha_k.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/fl_l1_grid/barplot_bandwidth_total_by_scenario.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/fl_l1_grid/macro_f1_by_round_grouped_by_alpha.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/fl_l1_grid/fpr_by_round_grouped_by_alpha.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/fl_l1_grid/p4_vs_all_p5_scenarios.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/fl_l1_grid/scenario_ranking_table.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/fl_l1_grid/heatmap_gap_macro_f1_vs_p4.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/fl_l1_grid/heatmap_bandwidth_alpha_k.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/fl_l1_grid/best_round_alpha_k.png`

## 7. Conclusion
La grid P5.3 confirme que FedAvg fonctionne très bien lorsque les données sont modérément hétérogènes ou quasi-IID. En revanche, FedAvg devient fragile en non-IID extrême, surtout lorsque le nombre de clients augmente.

Le scénario `alpha=5.0, K=3` est retenu comme meilleur scénario global pour FedAvg L1, car il maximise les performances tout en minimisant le coût de communication. Le scénario `alpha=0.5, K=5` reste un scénario réaliste non-IID performant à conserver pour les comparaisons avancées.

Cette limite de FedAvg en non-IID extrême justifie les prochaines phases du projet : Multi-tier/HeteroFL, QIFA et Quantum-Inspired optimization.
