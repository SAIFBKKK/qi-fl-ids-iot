# Model Factory 30 Rounds Report

Generated at: `2026-05-02T16:05:07.839799Z`

## 1. Resume executif

Trois modeles IDS offline ont ete entraines avec la pipeline `experiments/fl-iot-ids-v3` existante: `weak`, `medium`, `powerful`. Les runs utilisent Flower/FedAvg, 30 rounds, 3 clients, class weights, 28 features et 34 classes.

## 2. Fonctionnement actuel de fl-iot-ids-v3

La pipeline existante splitte les lignes raw avant preprocessing, fit le scaler sur `train`, cree des partitions FL par scenario, entraine avec `src.fl.client_app`/`src.fl.server_app`, evalue sur `val_preprocessed.npz`, puis sauvegarde checkpoints et summaries via `BaselineArtifactTracker`.

## 3. Scripts reutilises

`generate_scenarios.py`, `prepare_partitions.py`, `generate_weights.py`, `client_app.py`, `server_app.py`, `reporting_strategy.py`, `evaluate.py`, `losses.py`, `dataloader.py`, `artifact_logger.py`.

## 4. Changements effectues

Ajout de `configs/model_factory_30rounds.yaml`, `scripts/run_model_factory.py`, `scripts/validate_model_factory_bundles.py`; generalisation minimale de `MLPClassifier` aux listes de couches cachees.

## 5. Split 70/15/15

Scenario: `normal_noniid`. Rows: train=6580944, validation=1410203, deployment=1410203.

## 6. Verification anti data leakage

Le scaler declare `fit_split=train`; les partitions FL proviennent uniquement de `train_preprocessed.npz`; l'evaluation Flower utilise `val_preprocessed.npz`; `deployment_15.parquet` est exporte depuis `test.csv` uniquement et n'est jamais charge par le training.

## 7. Architectures

- weak: 28 -> 64 -> 34
- medium: 28 -> 128 -> 64 -> 34
- powerful: 28 -> 256 -> 128 -> 34

## 8. Configuration FL

FedAvg, 30 rounds, 1 local epoch, class weights actives, 3 clients.

## 9-10. Metriques validation et comparaison

| model | accuracy | macro-F1 | weighted-F1 | benign recall | attack recall | FPR | taille |
|---|---:|---:|---:|---:|---:|---:|---:|
| powerful | 0.816428 | 0.774383 | 0.812706 | 0.898356 | 0.973333 | 0.101644 | 177.5 KB |

Latence estimee: non mesuree dans ce run offline.

## 11. Emplacement des bundles

`C:\Users\saifb\dev\qi-fl-ids-iot\experiments\fl-iot-ids-v3\outputs\model_factory_30rounds`

## 12. Utilisation future dans Mode A

Chaque sous-dossier contient `global_model.pth`, `scaler.pkl`, `feature_names.pkl`, `label_mapping.json`, `model_config.json`, `run_summary.json`, `metrics.json`. Mode A pourra charger le bundle choisi, reconstruire `MLPClassifier` avec `hidden_layers`, appliquer `scaler.pkl` dans l'ordre `feature_names.pkl`, puis mapper la prediction avec `label_mapping.json`.

## 13. Limitations et prochaines etapes

Le dataset source externe n'etait pas visible ici; le run s'appuie donc sur les splits `normal_noniid` deja prepares et valides. Prochaine etape: brancher explicitement le choix weak/medium/powerful dans Mode A sans modifier les garanties anti-leakage.
