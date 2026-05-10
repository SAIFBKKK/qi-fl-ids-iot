# P6 - Hierarchical L2/L3 Flower Plan

## 1. P6.1 - L2 Family Flower FL

- Task : `l2_family`
- Dataset : `outputs/preprocessed/l2_family`
- Partitions : P3 `outputs/partitions/l2_family`, mode `index_only`
- Target : `y_family`
- Classes : 8 familles finales
- Modèle : `28 -> 128 -> 64 -> 8`
- Runtime : vrai Flower legacy/manual/subprocess, compatible Windows
- Métriques : accuracy, macro-F1, weighted-F1, per-family precision/recall/F1, confusion matrix 8x8, one-vs-rest TP/FP/TN/FN.

## 2. P6.2 - L3 Attack-Type Flower FL

- Task : `l3_attack_type`
- Dataset : mêmes features scaled L2
- Target : `label_id_original` remappé vers 33 classes attaque `0..32`
- BenignTraffic exclu car L2 est attack-only
- Modèle : `28 -> 128 -> 64 -> 33`
- Partitions : mêmes row_id P3 L2
- Métriques : macro-F1, weighted-F1, per-attack-type metrics, one-vs-rest, confusion matrix 33x33 et top confusion pairs.

## 3. P6.3 - Rapport hiérarchique

Le rapport P6 compare :

- L1 : modèle production/dashboard;
- L2 : familles d'attaque, expérimental;
- L3 : types d'attaque, expérimental.

L2/L3 ne sont pas intégrés au dashboard final.

## 4. Scénarios

Scénario principal full manuel :

- alpha = 0.5
- K = 3
- rounds = 30

Smoke autorisé :

- alpha = 0.5
- K = 3
- rounds = 1
- max_samples_per_client = 1000

## 5. Architecture runtime

Créer `src/fl_hierarchical/` avec :

- data loader index_only;
- modèles MLP configurables;
- métriques multiclasses et one-vs-rest;
- stratégie Flower FedAvg;
- runtime/manual/subprocess;
- plotting;
- summary schema;
- verify setup.

## 6. Artefacts

Chaque run écrit :

- checkpoints best/last;
- metrics CSV round/client/bandwidth/aggregation;
- metrics_val/test JSON;
- classification report JSON;
- confusion matrix CSV;
- one-vs-rest CSV;
- model config/class mapping;
- run_manifest.json;
- run_summary.json;
- logs Flower;
- latest_run.json et latest_run_summary.json.

## 7. Validation

Codex peut lancer :

- py_compile;
- tests légers;
- verify setup;
- smoke L2 1 round;
- smoke L3 1 round si RAM disponible.

Full L2/L3 30 rounds reste manuel.
