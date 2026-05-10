# P5 Reuse Notes — v3 to final FedAvg L1

## Fichiers v3 inspectés
- `experiments/fl-iot-ids-v3/src/fl/reporting_strategy.py`
- `experiments/fl-iot-ids-v3/src/fl/server_app.py`
- `experiments/fl-iot-ids-v3/src/fl/client_app.py`
- `experiments/fl-iot-ids-v3/src/fl/strategy.py`
- `experiments/fl-iot-ids-v3/src/fl/aggregation_hooks.py`
- `experiments/fl-iot-ids-v3/src/fl/metrics.py`
- `experiments/fl-iot-ids-v3/src/model/train.py`
- `experiments/fl-iot-ids-v3/src/model/evaluate.py`
- `experiments/fl-iot-ids-v3/configs/fl/fedavg_30rounds.yaml`

## Réutilisé conceptuellement
- Agrégation FedAvg pondérée par `num_examples`.
- Séparation client/server.
- Logs par round et par client.
- Suivi des coûts de communication à partir de la taille du modèle.
- Pattern de métriques agrégées et d'artefacts CSV/JSON.

## Réimplémenté pour le dossier final
- Code in-process sans dépendance runtime Flower obligatoire pour le mode verify/smoke.
- Modèle L1 binaire `28 -> 128 -> 64 -> 2` aligné avec P4.
- Chargement direct des partitions P3 finales `l1_binary/alpha_x/kY`.
- Protection explicite du global test holdout.
- Comparaison P4 vs P5 via les artefacts P4 finalisés.

## Non repris
- QIFA/QIFA-guard, SCAFFOLD, FedProx, multi-tier et node profiling v3.
- Dépendances de chemins v3 et logique 34 classes.
- Flower ServerApp/ClientApp runtime lourd, réservé à une éventuelle intégration ultérieure.

## Pourquoi
P5 doit être une baseline FedAvg L1 claire, reproductible et compatible avec les artefacts finaux P2/P3/P4, sans importer la complexité expérimentale v3 non nécessaire.
