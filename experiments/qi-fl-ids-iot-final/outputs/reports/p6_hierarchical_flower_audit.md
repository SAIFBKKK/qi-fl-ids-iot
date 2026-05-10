# P6 - Hierarchical L2/L3 Flower Audit

## 1. Fichiers audités

### Données et mappings finaux

- `outputs/preprocessed/l2_family/train_scaled.npz`
- `outputs/preprocessed/l2_family/val_scaled.npz`
- `outputs/preprocessed/l2_family/test_scaled.npz`
- `outputs/preprocessed/l2_family/manifest.json`
- `outputs/preprocessed/l2_family/family_mapping.json`
- `outputs/preprocessed/l2_family/distribution_report.json`
- `outputs/artifacts/mappings/label_mapping.json`
- `outputs/artifacts/mappings/id_to_label.json`
- `outputs/artifacts/mappings/label_to_family.json`
- `outputs/artifacts/mappings/label_to_binary.json`
- `outputs/artifacts/features/feature_names.json`
- `outputs/artifacts/scalers/l2_family_robust_scaler.pkl`

### Partitions P3 L2

Les 9 scénarios `outputs/partitions/l2_family/alpha_{0.1,0.5,5.0}/k{3,4,5}` existent. Ils sont en mode `index_only` avec `train_row_ids.npy` et `val_row_ids.npy` par client. Le global test holdout n'est pas partitionné.

### Runtime Flower final

P5.2/P5.2.1/P5.2.2 fournit le pattern à réutiliser :

- run id `run_YYYYMMDD_HHMMSS`;
- répertoires `runs/{run_id}/checkpoints`, `artifacts`, `logs`;
- `latest_run.json` et `latest_run_summary.json`;
- lancement manuel/subprocess Windows fiable;
- logs `flower_server.log`, `flower_clients.log`, `run_console.log`;
- contrat `run_summary.json` riche.

### Anciennes expériences hiérarchiques

Des anciens scripts hiérarchiques existent sous `experiments/baseline-CIC_IOT_2023/` et `experiments/fl-iot-ids-v2/`.

Points utiles :

- ancienne séparation L1/L2/L3;
- L2 family classification;
- L3 specialists et attack subtype datasets;
- configuration v2 avec `level2.output_dim=7`, désormais obsolète car le mapping final P2 contient 8 familles.

Ces éléments servent de référence conceptuelle uniquement. P6 doit utiliser les artefacts finaux P2/P3 et le runtime Flower final.

## 2. Données L2 disponibles

Le dataset L2 est attack-only :

- train : 6,370,944 lignes;
- validation : 1,365,202 lignes;
- test : 1,365,204 lignes;
- 28 features;
- target L2 : `y_family`;
- metadata conservée : `label_id_original`, `row_id`.

Le mapping famille final contient 8 familles :

- BruteForce;
- DDoS;
- DoS;
- Malware;
- Mirai;
- Recon;
- Spoofing;
- Web-based.

## 3. Possibilité de dériver L3

L3 peut être dérivé depuis `label_id_original` dans les NPZ L2. Comme L2 est attack-only, `BenignTraffic` est absent. Les 33 ids d'attaque originaux peuvent être remappés vers `0..32` pour un modèle `28 -> 128 -> 64 -> 33`.

Le mapping source `id_to_label.json` confirme 34 classes globales avec `BenignTraffic=1`. P6 L3 exclut `label_id=1`.

## 4. Partitions L2 index_only

Les partitions P3 L2 ne matérialisent pas de NPZ client. Chaque client référence uniquement :

- `train_row_ids.npy`;
- `val_row_ids.npy`.

P6 doit donc sélectionner les exemples dans les NPZ L2 globaux à partir de `row_id`, sans créer de gros datasets dupliqués.

## 5. Baseline Kaggle L3 historique

La baseline historique 34 classes est documentée dans `outputs/centralized_l1/artifacts/historical_kaggle_34class_baseline.json` :

- architecture : `28 -> 128 -> 64 -> 34`;
- best val macro-F1 : 0.811084;
- test macro-F1 : 0.810976;
- test accuracy : 0.841892.

Cette baseline est une référence historique L3 multiclasses. Elle n'est pas réentraînée et ne remplace pas P6 Flower.

## 6. Risques mémoire/performance

- Les NPZ L2 sont volumineux : train environ 790 MB, val/test environ 169 MB chacun.
- Un full Flower manuel avec clients séparés peut charger plusieurs fois les données globales si chaque client lit les NPZ. Cela peut consommer beaucoup de RAM.
- Les partitions `index_only` évitent la duplication disque, mais nécessitent une résolution `row_id -> position`.
- Smoke doit utiliser `max_samples_per_client=1000` pour rester léger.
- Full L2/L3 doit rester manuel et surveillé.

## 7. Recommandation P6

Implémenter P6 comme runtime Flower expérimental basé sur P5.2.2 :

- L2 Flower FL : 8 familles, target `y_family`;
- L3 Flower FL : 33 attaques, target dérivé de `label_id_original`;
- scénario principal : `alpha=0.5`, `K=3`, 30 rounds;
- smoke : 1 round, 1000 samples/client;
- L2/L3 non déployés dashboard;
- run summaries et figures riches pour rapport scientifique.
