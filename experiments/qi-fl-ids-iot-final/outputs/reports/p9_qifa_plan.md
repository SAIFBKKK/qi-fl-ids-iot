# P9 QIFA Plan

## P9.1 — QIFA Strategy Code-ready
- Créer `src/qifa/`.
- Implémenter `score -> theta -> amplitude -> probability -> omega`.
- Utiliser `omega = (1-gamma) * fedavg_weight + gamma * probability`.

## P9.2 — Verify + Smoke
- Vérifier les entrées P5/P8 et les partitions P3.
- Vérifier que `test_sent_to_clients=false`.
- Vérifier que les CSV QIFA sont générés.
- Lancer un smoke Flower 1 round.

## P9.3 — Full QIFA L1
- Scénario principal : `alpha=0.5`, `K=3`, `rounds=30`.
- Variante par défaut : `hybrid`.
- Gamma par défaut : `0.5`.

## P9.4 — Robustness
- Supporter `alpha in {0.1, 0.5, 5.0}`, `K=3`.
- Préparer un script de robustesse, sans exécution automatique.

## P9.5 — QIFA + QGA
- Option `--use-qga-mask`.
- Charger uniquement `final_selected_mask`.
- Forcer `selected_mask_id=conservative_seed_42` et `features_count=12`.

## P9.6 — Ablation finale
- Comparer :
- `P5 FedAvg baseline`
- `P8 FedAvg + QGA Flower`
- `P9 QIFA Flower`
- `P9 QIFA + QGA Flower`
