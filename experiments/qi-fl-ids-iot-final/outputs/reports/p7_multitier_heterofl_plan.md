# P7 - Multi-tier HeteroFL Plan

## 1. P7.1 - Multi-tier L1 HeteroFL

- Tâche : `l1_binary`
- Données : `outputs/partitions/l1_binary/`
- Cible : `y_binary`
- Output dim : 2
- Scénario principal : alpha=0.5, K=3, rounds=30
- Smoke : 1 round, 1000 samples/client

## 2. P7.2 - Multi-tier L2 HeteroFL

- Tâche : `l2_family`
- Données : `outputs/partitions/l2_family/` avec `index_only`
- Cible : `y_family`
- Output dim : auto depuis `family_mapping.json`
- Pas de L3 en P7
- Même scénario principal que L1

## 3. Tiers

K=3 : client_1=weak, client_2=medium, client_3=powerful.

K=4 : client_1=weak, client_2=weak, client_3=medium, client_4=powerful.

K=5 : client_1=weak, client_2=weak, client_3=medium, client_4=medium, client_5=powerful.

Cette répartition reflète un réseau IoT avec plus de nœuds faibles que de nœuds puissants.

## 4. HeteroFL slicing

Le serveur maintient le supernet maximal `28 -> 256 -> 128 -> output_dim`.

Chaque client reçoit un slice :

- weak : `fc1[0:64]` puis head `fc3[:,0:64]`
- medium : `fc1[0:128]`, `fc2[0:64,0:128]`, `fc3[:,0:64]`
- powerful : supernet complet

L'agrégation est une moyenne pondérée par slice. Une position non mise à jour conserve sa valeur globale.

## 5. Runtime

P7 utilise un runtime in-process contrôlé, cohérent avec P5 grid, pour faciliter la grid scientifique. Le runtime Flower démonstratif reste P5.2/P6.

## 6. Artefacts

Chaque run écrit checkpoints, métriques rounds/clients/tiers, bandwidth par tier, slices updated, metrics test, comparaison P4/P5/P6, figures et `run_summary.json`.

## 7. Commandes

Verify :

`python experiments/qi-fl-ids-iot-final/src/scripts/07_verify_multitier_heterofl_setup.py --config experiments/qi-fl-ids-iot-final/configs/multitier_heterofl.yaml`

Smoke L1 :

`python experiments/qi-fl-ids-iot-final/src/scripts/07_run_multitier_heterofl.py --config experiments/qi-fl-ids-iot-final/configs/multitier_heterofl.yaml --task l1 --mode smoke --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000`

Smoke L2 :

`python experiments/qi-fl-ids-iot-final/src/scripts/07_run_multitier_heterofl.py --config experiments/qi-fl-ids-iot-final/configs/multitier_heterofl.yaml --task l2 --mode smoke --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000`

Full principal L1/L2 : mêmes commandes avec `--mode full --rounds 30`.

Grid séquentielle : `powershell -ExecutionPolicy Bypass -File experiments/qi-fl-ids-iot-final/scripts/run_p7_multitier_grid_sequential.ps1 -Rounds 30`.

## 8. Critères d'acceptation code-ready

P7 est code-ready lorsque verify, tests, smoke L1 et smoke L2 passent, avec L3 exclu, dashboard/Docker/QI non modifiés et `global_test_holdout_protected=true`.
