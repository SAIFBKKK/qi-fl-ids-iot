# P7 - Multi-tier HeteroFL Audit

## 1. Fichiers audités

### P5 L1 FedAvg in-process

- `experiments/qi-fl-ids-iot-final/src/fl_l1/`
- `experiments/qi-fl-ids-iot-final/outputs/fl_l1_fedavg/`
- `experiments/qi-fl-ids-iot-final/outputs/reports/p5_grid_summary.csv`
- `experiments/qi-fl-ids-iot-final/outputs/reports/p5_grid_summary.json`
- `experiments/qi-fl-ids-iot-final/outputs/figures/fl_l1_grid/`

### P5.2 Flower L1 runtime

- `experiments/qi-fl-ids-iot-final/src/fl_l1_flower/`
- `experiments/qi-fl-ids-iot-final/outputs/fl_l1_flower/`

### P6 hierarchical Flower

- `experiments/qi-fl-ids-iot-final/src/fl_hierarchical/`
- `experiments/qi-fl-ids-iot-final/outputs/hierarchical_flower/`
- `experiments/qi-fl-ids-iot-final/outputs/preprocessed/l2_family/`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l2_family/`

### Configurations

- `configs/fl_l1_fedavg.yaml`
- `configs/fl_l1_flower.yaml`
- `configs/hierarchical_flower.yaml`
- `configs/tiers.yaml`

### Anciens composants multi-tier

- `experiments/fl-iot-ids-v3/src/model/supernet.py`
- `experiments/fl-iot-ids-v3/src/fl/masked_aggregation.py`
- `experiments/fl-iot-ids-v3/src/fl/node_profiler.py`
- `experiments/fl-iot-ids-v3/configs/fl/multitier.yaml`
- `experiments/fl-iot-ids-v3/configs/model/supernet_34.yaml`
- `experiments/fl-iot-ids-v3/tests/test_supernet.py`
- `experiments/fl-iot-ids-v3/tests/test_masked_aggregation.py`

## 2. Composants réutilisables

P5 fournit le runtime in-process rapide, les partitions L1 et les rapports de comparaison avec P4. P6 fournit le chargement L2 `index_only`, les métriques multiclasses, les figures et le contrat `run_summary.json`.

La v3 contient déjà la logique de base HeteroFL : extraction de sous-modèles par largeur, agrégation masquée pondérée et conservation des slices non mises à jour.

## 3. Limites FedAvg homogène

FedAvg P5 suppose que tous les clients entraînent le même modèle `28 -> 128 -> 64 -> 2`. Cette hypothèse ignore les contraintes IoT : RAM limitée, CPU faibles, latence et coûts de communication variables.

## 4. Contraintes HeteroFL

Le serveur doit conserver un supernet maximal. Les clients reçoivent seulement un slice adapté à leur tier. L'agrégation doit être faite par position de tenseur et pondérée par `num_examples`; les positions non couvertes par un tier doivent conserver l'ancienne valeur globale.

## 5. Modèles et tiers existants

La v3 utilisait des largeurs `weak=0.25`, `medium=0.5`, `powerful=1.0` sur un supernet `28 -> 256 -> 128 -> 34`.

Pour P7 final :

- weak : `28 -> 64 -> output_dim`
- medium : `28 -> 128 -> 64 -> output_dim`
- powerful : `28 -> 256 -> 128 -> output_dim`
- supernet : `28 -> 256 -> 128 -> output_dim`

## 6. Stratégie recommandée

Porter la logique v3 en package final `src/multitier_heterofl/`, en mode in-process contrôlé pour la grid scientifique. L1 utilise les partitions matérialisées P3. L2 réutilise les partitions P3 `index_only` et le loader P6.

## 7. Risques techniques

- L2 complet reste volumineux; les full runs L2 doivent rester manuels.
- Le tier weak saute la seconde couche cachée; l'agrégation doit donc ignorer `fc2` pour weak et conserver les slices non couvertes.
- Les comparaisons P4/P5/P6 sont des comparaisons contextuelles, pas toujours isométriques entre tâches.
- Les smoke runs à 1000 samples/client valident le pipeline, pas la performance scientifique.
