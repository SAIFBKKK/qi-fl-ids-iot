# P0 Audit Reuse Map

## Sommaire

[Phase P0 - Audit termine]

## Section 1 - Inventaire dataset

| Fichier | Chemin | Taille | Format | Role |
| --- | --- | --- | --- | --- |
| `balancing_v3_fixed300k_balanced.parquet` | `data/balancing_v3_fixed300k_outputs/balancing_v3_fixed300k_balanced.parquet` | `975 147 066` octets, `9 401 350 x 29` | Parquet, 9 row groups | Dataset final prioritaire pour P1/P2, sans duplication. |
| `balancing_v3_fixed300k_balanced.csv` | `data/balancing_v3_fixed300k_outputs/balancing_v3_fixed300k_balanced.csv` | `2 396 801 931` octets, `9 401 350 x 29` | CSV | Copie tabulaire equivalente, utile pour controles ponctuels mais non prioritaire. |
| `label_mapping.json` | `data/balancing_v3_fixed300k_outputs/label_mapping.json` | `895` octets, `34` classes | JSON | Mapping canonique classe -> `label_id`. |

Features confirmees dans le parquet et le CSV :

1. `flow_duration`
2. `Header_Length`
3. `Protocol Type`
4. `Duration`
5. `Rate`
6. `fin_flag_number`
7. `syn_flag_number`
8. `rst_flag_number`
9. `psh_flag_number`
10. `ack_flag_number`
11. `ack_count`
12. `syn_count`
13. `fin_count`
14. `urg_count`
15. `rst_count`
16. `HTTP`
17. `HTTPS`
18. `DNS`
19. `SSH`
20. `TCP`
21. `UDP`
22. `ARP`
23. `ICMP`
24. `Tot sum`
25. `Min`
26. `Std`
27. `IAT`
28. `Number`
29. `label_id`

Le mapping `label_mapping.json` contient :

| Classe | label_id |
| --- | --- |
| `Backdoor_Malware` | `0` |
| `BenignTraffic` | `1` |
| `BrowserHijacking` | `2` |
| `CommandInjection` | `3` |
| `DDoS-ACK_Fragmentation` | `4` |
| `DDoS-HTTP_Flood` | `5` |
| `DDoS-ICMP_Flood` | `6` |
| `DDoS-ICMP_Fragmentation` | `7` |
| `DDoS-PSHACK_Flood` | `8` |
| `DDoS-RSTFINFlood` | `9` |
| `DDoS-SYN_Flood` | `10` |
| `DDoS-SlowLoris` | `11` |
| `DDoS-SynonymousIP_Flood` | `12` |
| `DDoS-TCP_Flood` | `13` |
| `DDoS-UDP_Flood` | `14` |
| `DDoS-UDP_Fragmentation` | `15` |
| `DNS_Spoofing` | `16` |
| `DictionaryBruteForce` | `17` |
| `DoS-HTTP_Flood` | `18` |
| `DoS-SYN_Flood` | `19` |
| `DoS-TCP_Flood` | `20` |
| `DoS-UDP_Flood` | `21` |
| `MITM-ArpSpoofing` | `22` |
| `Mirai-greeth_flood` | `23` |
| `Mirai-greip_flood` | `24` |
| `Mirai-udpplain` | `25` |
| `Recon-HostDiscovery` | `26` |
| `Recon-OSScan` | `27` |
| `Recon-PingSweep` | `28` |
| `Recon-PortScan` | `29` |
| `SqlInjection` | `30` |
| `Uploading_Attack` | `31` |
| `VulnerabilityScan` | `32` |
| `XSS` | `33` |

## Section 2 - Mapping des fichiers réutilisables

| Phase | Fichier source à réutiliser | Chemin exact | Statut | Action prévue |
| --- | --- | --- | --- | --- |
| P1 - Data validation | `validate_data_pipeline.py`, `preflight.py`, `test_preprocessor.py` | `experiments/fl-iot-ids-v3/src/scripts/validate_data_pipeline.py`; `experiments/fl-iot-ids-v3/src/common/preflight.py`; `experiments/fl-iot-ids-v3/tests/test_preprocessor.py` | à adapter | Reprendre les controles de schema, classes, artefacts et invariants, puis les appliquer au parquet final `9 401 350 x 29`. |
| P2 - Preprocessing train-only | `generate_scenarios.py`, `fit_global_scaler.py`, `preprocess_node_data.py`, `preprocessor.py` | `experiments/fl-iot-ids-v3/src/scripts/generate_scenarios.py`; `experiments/fl-iot-ids-v3/src/scripts/fit_global_scaler.py`; `experiments/fl-iot-ids-v3/src/scripts/preprocess_node_data.py`; `experiments/fl-iot-ids-v3/src/data/preprocessor.py` | à adapter | Extraire uniquement le split + scaler train-only, retirer les scenarios v3 historiques et rebrancher les chemins finaux. |
| P3 - Dirichlet α∈{0.1,0.5,5.0} × K∈{3,4,5} | `prepare_partitions.py`, `generate_scenarios.py` | `experiments/fl-iot-ids-v3/src/scripts/prepare_partitions.py`; `experiments/fl-iot-ids-v3/src/scripts/generate_scenarios.py` | à étendre | Generaliser les splits Dirichlet sur toute la grille alpha/K avec manifests anti-leakage. |
| P4 - Centralized L1 baseline | `network.py`, `train.py`, `evaluate.py`, scripts baseline historiques | `experiments/fl-iot-ids-v3/src/model/network.py`; `experiments/fl-iot-ids-v3/src/model/train.py`; `experiments/fl-iot-ids-v3/src/model/evaluate.py`; `experiments/baseline-CIC_IOT_2023/src/training/train_hierarchical_level1_binary_v4.py` | à adapter | Reprendre MLP + boucle train/eval, reconstruire une baseline binaire L1 propre sur le dataset final. |
| P5 - FL L1 baseline | `reporting_strategy.py`, `server_app.py`, `client_app.py`, `strategy.py`, `aggregation_hooks.py`, configs FedAvg | `experiments/fl-iot-ids-v3/src/fl/reporting_strategy.py`; `experiments/fl-iot-ids-v3/src/fl/server_app.py`; `experiments/fl-iot-ids-v3/src/fl/client_app.py`; `experiments/fl-iot-ids-v3/src/fl/strategy.py`; `experiments/fl-iot-ids-v3/src/fl/aggregation_hooks.py`; `experiments/fl-iot-ids-v3/configs/fl/fedavg_30rounds.yaml` | à adapter | Garder le squelette Flower/FedAvg et remplacer les scenarios par les splits Dirichlet finaux. |
| P6 - Hierarchical L2/L3 experimental | Scripts hierarchical historiques | `experiments/baseline-CIC_IOT_2023/src/training/build_level2_family_dataset.py`; `experiments/baseline-CIC_IOT_2023/src/training/build_level3_subtype_datasets.py`; `experiments/baseline-CIC_IOT_2023/src/training/train_hierarchical_level2_family_v4.py`; `experiments/baseline-CIC_IOT_2023/src/training/train_hierarchical_level3_submodel_v4.py` | à adapter | Les sources L2/L3 ne sont pas dans `fl-iot-ids-v3/src`; les reprendre comme reference experimentale seulement. |
| P7 - Multi-tier HeteroFL | `supernet.py`, `node_profiler.py`, `multitier.yaml`, `tier_profiles.yaml`, model factory | `experiments/fl-iot-ids-v3/src/model/supernet.py`; `experiments/fl-iot-ids-v3/src/fl/node_profiler.py`; `experiments/fl-iot-ids-v3/configs/fl/multitier.yaml`; `experiments/fl-iot-ids-v3/configs/nodes/tier_profiles.yaml`; `experiments/fl-iot-ids-v3/scripts/run_model_factory.py` | à adapter | Reutiliser supernet et profils, revalider shared weights weak/medium/powerful dans la structure finale. |
| P8 - QGA feature selection | `qi_feature_selector.py`, `feature_selection.py`, `run_qi_feature_selection.py`, config QGA | `experiments/fl-iot-ids-v3/src/qi/qi_feature_selector.py`; `experiments/fl-iot-ids-v3/src/qi/feature_selection.py`; `experiments/fl-iot-ids-v3/src/scripts/run_qi_feature_selection.py`; `experiments/fl-iot-ids-v3/configs/qi/qga_feature_selection.yaml` | à adapter | Reprendre la logique QGA/QI FS, verrouiller l'usage sur tier powerful et documenter les masques. |
| P9 - QIFA aggregation | `qifa_strategy.py`, `qifa_guard_strategy.py`, configs QIFA | `experiments/fl-iot-ids-v3/src/fl/qifa_strategy.py`; `experiments/fl-iot-ids-v3/src/fl/qifa_guard_strategy.py`; `experiments/fl-iot-ids-v3/configs/fl/qifa_formula_30rounds.yaml`; `experiments/fl-iot-ids-v3/configs/fl/qifa_guard_30rounds.yaml` | à adapter | Conserver les formules testees, rebrancher sur les metrics finales et garder le cas `lambda_qifa=0` comme controle. |
| P10 - Robustness and poisoning attacks | Tests de strategies et aggregation | `experiments/fl-iot-ids-v3/tests/test_qifa_guard_strategy.py`; `experiments/fl-iot-ids-v3/tests/test_masked_aggregation.py`; `experiments/fl-iot-ids-v3/src/fl/masked_aggregation.py` | à étendre | Ajouter les attaques de poisoning et scenarios robustesse dans le nouveau dossier, a partir des invariants d'agregation deja testes. |
| P11 - FedTN/MPS compression | Aucun module FedTN/MPS trouve | `experiments/fl-iot-ids-v3/src/model/supernet.py` comme point d'ancrage seulement | non applicable | Implementer plus tard un module dedie dans `src/fl/compression/`; PowerSGD n'est pas retenu. |
| P12 - Ablation and evaluation reports | Scripts evaluation QI | `experiments/fl-iot-ids-v3/src/scripts/build_ablation_table.py`; `experiments/fl-iot-ids-v3/src/scripts/build_qi_benchmark_reduced_report.py`; `experiments/fl-iot-ids-v3/src/scripts/evaluate_confusion_matrices.py`; `experiments/fl-iot-ids-v3/src/scripts/evaluate_per_class_metrics.py` | à adapter | Reutiliser les generateurs de tableaux, confusion matrices et analyses per-class avec chemins finaux. |
| P13 - Dashboard L1 final | Dashboard et APIs services | `services/dashboard/main.py`; `services/dashboard/api/metrics.py`; `services/dashboard/api/nodes.py`; `services/dashboard/api/models.py`; `services/dashboard/api/qi.py`; `services/dashboard/static/app.js`; `services/dashboard/templates/tab_iot.html`; `services/dashboard/templates/tab_fl.html` | à adapter | Garder la structure UI/API, retirer les metriques QI attendues/fictives et brancher uniquement les mesures L1 finales. |
| P14 - Docker stack and final delivery | Compose, FL server, IoT node, monitoring | `services/docker-compose.yml`; `services/fl-server/server_entrypoint.py`; `services/iot-node/main.py`; `services/monitoring/prometheus.yml`; `services/monitoring/grafana/dashboards/05_fl_server.json`; `services/monitoring/grafana/dashboards/qi_fl_ids_overview.json` | à adapter | Reprendre la stack, ajuster les volumes/configs vers le dossier final et valider les profils K=3/4/5. |

## Section 3 - Cartographie complete

### 3.1 `experiments/fl-iot-ids-v3/src/common/`

| Fichier | Role | Decision | Justification |
| --- | --- | --- | --- |
| `experiments/fl-iot-ids-v3/src/common/__init__.py` | Package marker. | ignorer | Placeholder sans logique. |
| `experiments/fl-iot-ids-v3/src/common/config.py` | Chargement YAML, merge profond et bundles d'experiment. | adapter | Utile, mais les chemins/configs doivent pointer vers `qi-fl-ids-iot-final`. |
| `experiments/fl-iot-ids-v3/src/common/logger.py` | Creation de logger applicatif. | reutiliser | Peu couple au domaine, reutilisable avec noms de logger finaux. |
| `experiments/fl-iot-ids-v3/src/common/paths.py` | Resolution des chemins raw/processed et creation de dossiers runtime. | adapter | Chemins v3 a remplacer, risque de sorties hors dossier final. |
| `experiments/fl-iot-ids-v3/src/common/preflight.py` | Validation des artefacts requis avant run. | adapter | Bon modele pour P1/P5, mais la liste d'artefacts finaux change. |
| `experiments/fl-iot-ids-v3/src/common/registry.py` | Lecture du registre d'experiences v3. | adapter | Le registre final sera structure differemment. |
| `experiments/fl-iot-ids-v3/src/common/runtime.py` | Configuration des artefacts runtime/scaffold. | adapter | Necessaire seulement si SCAFFOLD ou stateful strategies sont retenues en final. |
| `experiments/fl-iot-ids-v3/src/common/schemas.py` | Schemas `NodeConfig`, `NodeProfile`, `TierAssignment`. | adapter | Base utile pour K=3/4/5 et tiers, mais les champs doivent etre stabilises. |
| `experiments/fl-iot-ids-v3/src/common/utils.py` | Seeds et resolution des IDs noeuds. | adapter | Utile, mais doit accepter K variable et nouveaux manifests. |

### 3.2 `experiments/fl-iot-ids-v3/src/data/`

| Fichier | Role | Decision | Justification |
| --- | --- | --- | --- |
| `experiments/fl-iot-ids-v3/src/data/__init__.py` | Package marker. | ignorer | Placeholder sans logique. |
| `experiments/fl-iot-ids-v3/src/data/collector.py` | Collecteur local minimal. | ignorer | Peu pertinent pour validation offline finale. |
| `experiments/fl-iot-ids-v3/src/data/dataloader.py` | Creation des dataloaders par noeud. | adapter | Reutilisable apres standardisation des chemins `.npz` finaux. |
| `experiments/fl-iot-ids-v3/src/data/dataset.py` | Dataset local IoT pour splits pretraites. | adapter | Bon point de depart pour P2/P5, a verifier avec L1 binaire. |
| `experiments/fl-iot-ids-v3/src/data/partitioning.py` | Manifest de partitions. | adapter | Fonction simple à étendre aux manifests Dirichlet finaux. |
| `experiments/fl-iot-ids-v3/src/data/preprocessor.py` | `BaselinePreprocessor`, scaling et transformation. | adapter | A conserver sous controle train-only et 28 features finales. |

### 3.3 `experiments/fl-iot-ids-v3/src/fl/`

| Fichier | Role | Decision | Justification |
| --- | --- | --- | --- |
| `experiments/fl-iot-ids-v3/src/fl/__init__.py` | Package marker. | ignorer | Placeholder sans logique. |
| `experiments/fl-iot-ids-v3/src/fl/aggregation_hooks.py` | Aggregation des metriques fit/evaluate. | reutiliser | Logique generique et testee, utile pour reporting final. |
| `experiments/fl-iot-ids-v3/src/fl/client_app.py` | Client Flower, chargement modele/donnees et feature selection. | adapter | Tres utile, mais fortement couple aux configs v3 et chemins d'artefacts. |
| `experiments/fl-iot-ids-v3/src/fl/masked_aggregation.py` | Aggregation de sous-tenseurs pour modeles heterogenes. | adapter | Base utile pour HeteroFL, a revalider avec tiers finaux. |
| `experiments/fl-iot-ids-v3/src/fl/metrics.py` | Moyenne ponderee de metriques. | reutiliser | Simple et reutilisable. |
| `experiments/fl-iot-ids-v3/src/fl/node_profiler.py` | Profilage des noeuds et assignation de tiers. | adapter | Pertinent P7/P14, mais profils materiels finaux a redefinir. |
| `experiments/fl-iot-ids-v3/src/fl/qifa_guard_strategy.py` | QIFA garde-fous, pondération qualite/rarete, clipping. | adapter | Logique centrale P9, a brancher sur runs finaux. |
| `experiments/fl-iot-ids-v3/src/fl/qifa_strategy.py` | QIFA formulee sur diversite des updates. | adapter | Logique centrale P9, documentee et testee. |
| `experiments/fl-iot-ids-v3/src/fl/reporting_strategy.py` | `ReportingFedAvg`, `ReportingScaffold`, checkpoints et rapports. | adapter | Pilier P5/P9, mais sorties v3 et MLflow a reconfigurer. |
| `experiments/fl-iot-ids-v3/src/fl/server_app.py` | Construction serveur Flower et strategies. | adapter | Reutilisable avec registry/config final. |
| `experiments/fl-iot-ids-v3/src/fl/strategy.py` | Factory de strategies. | adapter | À étendre pour FedAvg final, QIFA, QIFA guard et exclusions. |

### 3.4 `experiments/fl-iot-ids-v3/src/model/`

| Fichier | Role | Decision | Justification |
| --- | --- | --- | --- |
| `experiments/fl-iot-ids-v3/src/model/__init__.py` | Package marker. | ignorer | Placeholder sans logique. |
| `experiments/fl-iot-ids-v3/src/model/evaluate.py` | Evaluation accuracy, macro-F1, recall benign/rare et comptes classes. | adapter | Metrics IDS utiles, a aligner L1 binaire et rapports finaux. |
| `experiments/fl-iot-ids-v3/src/model/losses.py` | Focal loss et class weights. | adapter | Utile si class weights retenus, mais la baseline finale doit etre explicite. |
| `experiments/fl-iot-ids-v3/src/model/network.py` | `MLPClassifier` flat. | adapter | Base MLP pour L1 et flat 34 classes; sorties L1 a redefinir. |
| `experiments/fl-iot-ids-v3/src/model/supernet.py` | Supernet, extraction sous-modeles, comptage parametres. | adapter | Base principale pour P7 HeteroFL. |
| `experiments/fl-iot-ids-v3/src/model/train.py` | Boucles `train_one_epoch` et `train_local`. | adapter | Reutilisable avec dataloaders finaux. |
| `experiments/fl-iot-ids-v3/src/model/validation.py` | Validation dimensions modele/classes/width. | adapter | Utile pour eviter incoherences L1/L2/L3/tier. |

### 3.5 `experiments/fl-iot-ids-v3/src/qi/`

| Fichier | Role | Decision | Justification |
| --- | --- | --- | --- |
| `experiments/fl-iot-ids-v3/src/qi/__init__.py` | Package QI. | reutiliser | Declaration claire du scope quantum-inspired. |
| `experiments/fl-iot-ids-v3/src/qi/feature_selection.py` | QGA feature selection avec population, fitness et artefacts. | adapter | Utile pour P8, a synchroniser avec la selection finale et tier powerful. |
| `experiments/fl-iot-ids-v3/src/qi/qi_feature_selector.py` | Variante QI theta/probabilites, mini-MLP fitness et sauvegarde. | adapter | Module le plus complet pour P8, mais configs/smoke/full a clarifier. |

### 3.6 `experiments/fl-iot-ids-v3/src/scripts/`

| Fichier | Role | Decision | Justification |
| --- | --- | --- | --- |
| `experiments/fl-iot-ids-v3/src/scripts/__init__.py` | Package marker. | ignorer | Placeholder sans logique. |
| `experiments/fl-iot-ids-v3/src/scripts/build_ablation_table.py` | Construction de tables d'ablation. | adapter | Utile P12 avec nouveaux noms d'experiences. |
| `experiments/fl-iot-ids-v3/src/scripts/build_baseline_bundle.py` | Bundle deployment baseline FL. | adapter | Bon modele P13/P14, mais artefacts L1 finaux differents. |
| `experiments/fl-iot-ids-v3/src/scripts/build_qi_benchmark_reduced_report.py` | Rapport benchmark QI reduit et figures. | adapter | Utile P12, rebrancher sur resultats finaux. |
| `experiments/fl-iot-ids-v3/src/scripts/evaluate_confusion_matrices.py` | Confusion matrices et figures par experience. | adapter | Utile P4/P5/P12, dimensions/classes a ajuster L1. |
| `experiments/fl-iot-ids-v3/src/scripts/evaluate_per_class_metrics.py` | Metrics par classe, deltas QGA/QIFA, figures. | adapter | Utile pour 34 classes/L2-L3; pour L1, simplifier. |
| `experiments/fl-iot-ids-v3/src/scripts/fit_global_scaler.py` | Fit scaler global. | adapter | A reprendre seulement si le fit est strictement train-only. |
| `experiments/fl-iot-ids-v3/src/scripts/generate_scenarios.py` | Partition + preprocess scenarios v3. | adapter | Contient le bon pattern anti-leakage, mais scenarios non conformes au Dirichlet final. |
| `experiments/fl-iot-ids-v3/src/scripts/generate_weights.py` | Class weights depuis manifests. | adapter | Utile si class weights retenus en P4/P5. |
| `experiments/fl-iot-ids-v3/src/scripts/prepare_partitions.py` | Partition Dirichlet et stats noeuds. | à étendre | Point de depart direct pour alpha/K grid. |
| `experiments/fl-iot-ids-v3/src/scripts/preprocess_node_data.py` | Preprocess par noeud avec scaler existant. | adapter | Reutilisable apres verrouillage train-only. |
| `experiments/fl-iot-ids-v3/src/scripts/run_client.py` | Client FL script legacy. | adapter | Moins central que `client_app.py`, mais utile comme runbook. |
| `experiments/fl-iot-ids-v3/src/scripts/run_experiment.py` | Orchestration d'experience et MLflow. | adapter | Bon squelette, chemins et registry a refaire. |
| `experiments/fl-iot-ids-v3/src/scripts/run_qi_feature_selection.py` | CLI QGA/QI FS. | adapter | Utile P8. |
| `experiments/fl-iot-ids-v3/src/scripts/run_server.py` | Serveur FL legacy et tracking communication. | adapter | Reference utile, mais `server_app.py` est plus moderne. |
| `experiments/fl-iot-ids-v3/src/scripts/smoke_test.py` | Smoke test minimal. | ignorer | Trop superficiel pour validation finale. |
| `experiments/fl-iot-ids-v3/src/scripts/test_dataloader.py` | Script manuel dataloader. | ignorer | Remplacer par tests pytest finaux. |
| `experiments/fl-iot-ids-v3/src/scripts/test_local_training.py` | Script manuel training local. | adapter | Peut inspirer smoke P4/P5. |
| `experiments/fl-iot-ids-v3/src/scripts/validate_bundle.py` | Validation bundle deployment en 10 tests. | adapter | Tres utile P13/P14, a actualiser pour L1 final. |
| `experiments/fl-iot-ids-v3/src/scripts/validate_data_pipeline.py` | Validation basic stats/classes/imbalance par noeud. | adapter | Base P1/P3. |

### 3.7 `experiments/fl-iot-ids-v3/src/services/`

| Fichier | Role | Decision | Justification |
| --- | --- | --- | --- |
| `experiments/fl-iot-ids-v3/src/services/__init__.py` | Package marker. | ignorer | Placeholder sans logique. |
| `experiments/fl-iot-ids-v3/src/services/collector_service.py` | Stub service collector. | ignorer | Remplace par `services/iot-node` et `services/edge-ids-gateway`. |
| `experiments/fl-iot-ids-v3/src/services/fl_client_service.py` | Stub service client FL. | ignorer | Remplace par `services/fl-client` et stack Compose. |
| `experiments/fl-iot-ids-v3/src/services/preprocessor_service.py` | Stub service preprocessor. | ignorer | Remplace par preprocessing offline P2 et services gateway. |

### 3.8 `experiments/fl-iot-ids-v3/src/tracking/` et `src/utils/`

| Fichier | Role | Decision | Justification |
| --- | --- | --- | --- |
| `experiments/fl-iot-ids-v3/src/tracking/__init__.py` | Package tracking. | ignorer | Placeholder. |
| `experiments/fl-iot-ids-v3/src/tracking/artifact_logger.py` | Journalisation MLflow/artefacts, configs resolues, checkpoints. | adapter | Utile P12/P14, mais chemins absolus a eviter. |
| `experiments/fl-iot-ids-v3/src/tracking/run_naming.py` | Nommage des runs. | adapter | A aligner avec P1-P14. |
| `experiments/fl-iot-ids-v3/src/utils/mlflow_logger.py` | Logger MLflow utilitaire. | adapter | Optionnel, dependra du tracking final. |

### 3.9 Artefacts existants audites

Le chemin demande `experiments/fl-iot-ids-v3/processed/artifacts/` n'existe pas. Les artefacts reels observes sont :

| Zone | Chemins/fichiers | Decision |
| --- | --- | --- |
| Artefacts scaler/features/classes | `experiments/fl-iot-ids-v3/artifacts/scaler_standard_train_normal_noniid.pkl`, `scaler_standard_train_absent_local.pkl`, `scaler_standard_train_rare_expert.pkl`, `feature_names_*.pkl`, `class_weights_*.pkl` | Ne pas reutiliser comme source finale; utiliser comme reference de format. |
| Artefacts QGA | `experiments/fl-iot-ids-v3/artifacts/qi_feature_selection/normal_noniid/selected_features.json`, `feature_mask.npy`, `selection_report.md`; equivalents `absent_local` | Reutiliser comme reference P8, pas comme selection finale sans rerun. |
| Splits pretraites | `experiments/fl-iot-ids-v3/data/processed/{normal_noniid,absent_local,rare_expert}/node*/{train,val,test}_preprocessed.npz` | Reference de structure, mais scenarios non conformes a alpha/K finaux. |
| Manifests splits | `experiments/fl-iot-ids-v3/data/splits/*_manifest.json` | Reference de structure anti-leakage. |
| Bundle baseline deployment | `experiments/fl-iot-ids-v3/outputs/deployment/baseline_fedavg_normal_classweights/` | Reutiliser le format de bundle, pas les poids comme resultat final. |
| Model factory 30 rounds | `experiments/fl-iot-ids-v3/outputs/model_factory_30rounds/{weak,medium,powerful}/global_model.pth`, `model_config.json`, `metrics.json`, `run_summary.json` | Reference P7/P13; les configs confirment `input_dim=28`, `num_classes=34`. |
| Exported tier models | `experiments/fl-iot-ids-v3/outputs/exported_models/{weak,medium,powerful}/model.pth` | Reference de tailles weak/medium/powerful. |
| Rapports QI | `experiments/fl-iot-ids-v3/outputs/reports/qi_benchmark_reduced/` | Reutiliser les tableaux/figures comme modele de rapport, pas comme validation finale. |

Fichiers notables dans `outputs/reports/qi_benchmark_reduced/` :

- `final_report.md`
- `final_comparison_table.csv`
- `ablation_qga_effect.csv`
- `ablation_qifa_effect.csv`
- `ablation_qga_qifa_interaction.csv`
- `qga_selected_features_report.md`
- `qifa_internal_dynamics.csv`
- `run_status.csv`
- `figures/figure1_macro_f1_convergence.png` a `figure6_bandwidth.png`
- `confusion_matrices/E*/classification_report.json`, `confusion_matrix_raw.csv`, `confusion_matrix_normalized.csv`
- `per_class_metrics/per_class_analysis.md`, `per_class_comparison_all_experiments.csv`, deltas QGA/QIFA et figures associees

### 3.10 Microservices audites

| Service | Chemin | Role observe | Decision P13/P14 |
| --- | --- | --- | --- |
| FL server | `services/fl-server/` | FastAPI/Flower dispatcher, registry de noeuds/modeles, metrics Prometheus, assignation de tiers. | À adapter pour modeles L1 finaux, profils K=3/4/5 et chemins de bundles finaux. |
| IoT node | `services/iot-node/` | Collecte MQTT, preprocessing, inference API, registration et metrics noeud. | À adapter pour L1 binaire production et artefacts finaux. |
| Dashboard | `services/dashboard/` | UI FastAPI/Jinja/static, APIs nodes/models/metrics/QI/scenarios/system. | À adapter; retirer `services/dashboard/data/qi_metrics.yaml` car plusieurs valeurs sont `expected` et non mesurees. |
| Monitoring | `services/monitoring/` | Prometheus, Grafana dashboards IDS/FL/QI. | À adapter pour panels finaux et supprimer les panels QI non mesures du dashboard final. |
| QGA service | `services/qga-service/` | Optimiseur QGA expose en service HTTP avec metrics. | Reference possible P8/P14, mais P0 ne deplace rien. |
| Feature extractor | `services/feature-extractor/` | Extraction des 28 features CIC-IoT depuis evenements bruts. | Utile pour demo/inference, hors coeur offline P1-P12. |
| Edge IDS gateway | `services/edge-ids-gateway/` | Mapping raw event -> features, preprocessing, inference edge. | Utile P13/P14 si dashboard production s'appuie sur flux live. |
| Traffic generator | `services/traffic-generator/` | Replay de trafic et metrics. | Utile pour demo stack Docker. |

### 3.11 Tests existants utiles

| Zone | Tests | Reutilisation |
| --- | --- | --- |
| v3 dataset/preprocessing | `experiments/fl-iot-ids-v3/tests/test_dataset.py`, `test_preprocessor.py`, `test_run_preflight.py` | Modeles pour P1/P2. |
| v3 FL invariants | `test_fl_invariants.py`, `test_fl_smoke.py`, `test_masked_aggregation.py`, `test_node_profiler.py` | Modeles P5/P7/P10. |
| v3 model/QI | `test_model.py`, `test_supernet.py`, `test_qi_feature_selection.py`, `test_qi_feature_selector.py`, `test_qifa_strategy.py`, `test_qifa_guard_strategy.py` | Modeles P7/P8/P9. |
| v3 evaluation/tracking | `test_per_class_metrics.py`, `test_tracking.py` | Modeles P12. |
| services FL/Docker | `services/fl-server/tests/test_node_registry.py`, `test_tier_assignment.py` | Modeles P14. |
| services IoT | `services/iot-node/tests/test_collector.py`, `test_inference.py`, `test_node_registration.py`, `test_preprocessor.py` | Modeles P13/P14. |
| services gateway/extractor | `services/edge-ids-gateway/tests/*.py`, `services/feature-extractor/tests/test_extractor.py` | Modeles pour pipeline live/dashboard. |

### 3.12 Documentation existante utile

| Document | Role | Decision |
| --- | --- | --- |
| `docs/qi/qifa.md` | Definition QIFA, formule, controle FedAvg, metriques. | Reutiliser comme reference P9. |
| `docs/qi/qga_feature_selection.md` | Definition QGA/QI feature selection, representation theta, artefacts. | Reutiliser comme reference P8. |
| `experiments/fl-iot-ids-v3/docs/data_pipeline.md` | Notes pipeline data v3. | Adapter P1/P2. |
| `docs/reports/MODEL_FACTORY_30ROUNDS_REPORT.md` | Rapport model factory. | Reference P7/P12. |
| `docs/SCOPE_DECISIONS.md` | Decisions de scope existantes. | Lire avant P1 pour eviter divergences. |

## Section 4 - Decisions architecturales gelees

- Dataset : `balancing_v3_fixed300k_balanced.parquet` (`9 401 350 x 29`).
- 28 features + 1 `label_id`.
- Dirichlet α∈{0.1, 0.5, 5.0} × K∈{3, 4, 5}.
- L1 binaire = production et dashboard final.
- L2/L3 experimentaux = rapport seulement.
- Multi-tier weak/medium/powerful avec shared weights (HeteroFL).
- QI : QGA + QIFA + FedTN/MPS sur tier powerful uniquement.
- QIARM : future work uniquement.
- PowerSGD : non retenu, FedTN/MPS direct.

## Section 5 - Risques identifies

- `experiments/fl-iot-ids-v3/processed/artifacts/` est absent; les artefacts reels sont sous `experiments/fl-iot-ids-v3/artifacts/` et `experiments/fl-iot-ids-v3/data/processed/`.
- Plusieurs summaries v3 contiennent des chemins absolus Windows, par exemple dans `experiments/fl-iot-ids-v3/outputs/model_factory_30rounds/model_factory_summary.json`.
- Les scenarios v3 (`normal_noniid`, `absent_local`, `rare_expert`) ne correspondent pas a la matrice finale Dirichlet alpha/K.
- Les artefacts scaler/class weights v3 sont des references de format, mais ne doivent pas etre reutilises comme artefacts finaux sans regeneration train-only.
- Les modules v3 sont couples a `experiments/fl-iot-ids-v3` par les configs, registries et chemins runtime.
- Les sources hierarchical L2/L3 ne sont pas dans `experiments/fl-iot-ids-v3/src`; elles existent plutot dans `experiments/baseline-CIC_IOT_2023/src/training/`.
- `services/dashboard/data/qi_metrics.yaml` contient des valeurs `expected` issues de litterature ou compositions conservatrices; elles doivent etre retirees du dashboard final mesure.
- Aucun module FedTN/MPS dedie n'a ete trouve dans v3; P11 demandera une implementation nouvelle apres validation.
- Aucun module d'attaques dedie n'a ete trouve dans v3; P10 devra etre cree apres validation.
- `services/qga-service/.venv/` existe dans l'arborescence locale et peut polluer les recherches/tests si les exclusions ne sont pas explicites.
- Des `__pycache__` et caches pytest existent dans plusieurs dossiers historiques; ne pas les prendre comme artefacts scientifiques.
- Les bundles model factory contiennent `num_classes=34`; la production L1 binaire devra redefinir clairement la sortie modele et les mappings de decision.

Commandes de reproduction de l'audit :

PowerShell Windows :

```powershell
Get-ChildItem -Force data\balancing_v3_fixed300k_outputs
@'
from pathlib import Path
import json
import pyarrow.parquet as pq
p = Path("data/balancing_v3_fixed300k_outputs/balancing_v3_fixed300k_balanced.parquet")
pf = pq.ParquetFile(p)
print(pf.metadata.num_rows, pf.metadata.num_columns)
print(pf.schema_arrow.names)
print(json.load(open("data/balancing_v3_fixed300k_outputs/label_mapping.json", encoding="utf-8")))
'@ | python -
Get-ChildItem -Recurse -File experiments\fl-iot-ids-v3\src
Get-ChildItem -Recurse -File experiments\fl-iot-ids-v3\artifacts
Get-ChildItem -Recurse -File services\fl-server
Get-ChildItem -Recurse -File services\dashboard
git status --short
```

Linux/macOS bash :

```bash
ls -lh data/balancing_v3_fixed300k_outputs
python - <<'PY'
from pathlib import Path
import json
import pyarrow.parquet as pq
p = Path("data/balancing_v3_fixed300k_outputs/balancing_v3_fixed300k_balanced.parquet")
pf = pq.ParquetFile(p)
print(pf.metadata.num_rows, pf.metadata.num_columns)
print(pf.schema_arrow.names)
print(json.load(open("data/balancing_v3_fixed300k_outputs/label_mapping.json", encoding="utf-8")))
PY
find experiments/fl-iot-ids-v3/src -type f | sort
find experiments/fl-iot-ids-v3/artifacts -type f | sort
find services/fl-server -type f | sort
find services/dashboard -type f | sort
git status --short
```

## Section 6 - Prochaine etape

Tags de suivi : utiliser uniquement le préfixe `final-v`, par exemple `final-v0.1.1-skeleton-cleanup` si un tag P0.1 est créé après revue.

P0 se termine ici. Le dossier `experiments/qi-fl-ids-iot-final/` contient uniquement le skeleton, les placeholders et ce rapport d'audit.

P1 (Data Validation) doit etre validee explicitement par l'utilisateur avant toute implementation. Aucune logique de validation, preprocessing, training, FL, QI, dashboard ou Docker final n'a ete codee pendant P0.
