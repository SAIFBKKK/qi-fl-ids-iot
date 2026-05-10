# P9 QIFA Audit

## Fichiers inspectés
- `experiments/qi-fl-ids-iot-final/configs/fl_l1_fedavg.yaml`
- `experiments/qi-fl-ids-iot-final/configs/fl_l1_flower.yaml`
- `experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml`
- `experiments/qi-fl-ids-iot-final/src/fl_l1/`
- `experiments/qi-fl-ids-iot-final/src/fl_l1_flower/`
- `experiments/qi-fl-ids-iot-final/src/qga/flower_runtime.py`
- `experiments/qi-fl-ids-iot-final/outputs/reports/p5_grid_summary.csv`
- `experiments/qi-fl-ids-iot-final/outputs/reports/p5_grid_comparative_table.md`
- `experiments/qi-fl-ids-iot-final/outputs/reports/p8_qga_ablation_summary.csv`
- `experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/final_selected_mask/selection_decision.json`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l1_binary/alpha_0.1/k3`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l1_binary/alpha_0.5/k3`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l1_binary/alpha_5.0/k3`

## Constats
- La baseline P5 FedAvg L1 est disponible avec une grille complète `alpha x K`.
- Le runtime Flower L1 P5.2/P5.2.1 est déjà fonctionnel avec `run_id`, logs serveur/client, smoke subprocess et `latest_run_summary.json`.
- Le masque QGA calibré P8 est disponible et figé : `selected_mask_id=conservative_seed_42`, `features_count=12`.
- Les partitions L1 P3 protègent déjà le test global holdout et exposent `train_scaled.npz` / `val_scaled.npz` par client.

## Risques QIFA
- Les scores QIFA peuvent devenir numériquement instables si tous les clients ont des métriques très proches.
- Un `gamma` trop élevé peut surpondérer un client et nuire à la stabilité en non-IID extrême.
- La pénalité de drift doit rester interprétable et non dominante, sinon QIFA peut se réduire à un simple filtre de divergence.
- Le logging doit être assez riche pour permettre d’expliquer les poids finaux par round.

## Recommandation d’architecture
- Réutiliser le runtime Flower P5.2 comme base.
- Implémenter QIFA dans une stratégie Flower dédiée qui surcharge `aggregate_fit`.
- Garder la même ergonomie `verify`, `smoke`, `server`, `client`.
- Ajouter une option `--use-qga-mask` pour séparer proprement `QIFA` et `QIFA + QGA`.
