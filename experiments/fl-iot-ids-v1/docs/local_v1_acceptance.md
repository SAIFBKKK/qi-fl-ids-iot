# Local V1 Acceptance

## Objet
Ce document formalise l’acceptation fonctionnelle de la V1 locale du système de Federated Learning pour IDS IoT.

## Périmètre de la V1
La V1 couvre :
- un serveur central Flower
- trois nœuds IoT simulés
- un pipeline local complet :
  - partitionnement des données
  - preprocessing compatible baseline
  - chargement PyTorch
  - entraînement local
  - exécution fédérée

La V1 a été conçue comme une base minimale, modulaire et reproductible, avant l’ajout de composants plus avancés comme EdgeX, MQTT, observabilité complète ou orchestration edge réelle.

## Références baseline
La cohérence avec la baseline centralisée est assurée à partir des artifacts suivants :
- `artifacts/feature_names.pkl`
- `artifacts/scaler_robust.pkl`
- `artifacts/label_mapping_34.pkl`
- `artifacts/class_weights_34.pkl`

## Critères d’acceptation fonctionnelle

### Données
- [x] Le partitionnement du dataset en 3 nœuds fonctionne
- [x] Un manifeste de partitionnement est généré
- [x] Chaque nœud possède son dataset local brut
- [x] Le preprocessing local recharge correctement les artifacts baseline
- [x] L’ordre des features est conforme à la baseline
- [x] Le scaling est cohérent avec la baseline
- [x] Le mapping des labels est cohérent avec la baseline
- [x] Les fichiers `.npz` locaux sont générés correctement

### Couche ML locale
- [x] Le `DataLoader` PyTorch fonctionne sur les données locales
- [x] Le modèle MLP local s’initialise correctement
- [x] L’entraînement local mono-client fonctionne
- [x] La loss décroît de manière cohérente
- [x] L’accuracy locale progresse de manière cohérente

### Couche FL
- [x] Le serveur Flower fonctionne
- [x] Les clients Flower fonctionnent
- [x] Le scénario 1 serveur + 3 clients a été exécuté avec succès
- [x] 3 rounds ont été exécutés sans échec
- [x] L’agrégation `fit` a reçu 3 résultats sur 3
- [x] L’agrégation `evaluate` a reçu 3 résultats sur 3
- [x] Les métriques distribuées montrent une amélioration globale

## Résultats observés

### Validation locale explicite
Le mode local explicite avec :
- `python -m src.scripts.run_server`
- `python -m src.scripts.run_client`

a permis de valider :
- 3 clients sur 3 disponibles
- 3 rounds exécutés
- 0 failure côté agrégation
- baisse de la loss distribuée
- hausse de l’accuracy distribuée

### Validation Flower moderne
Le mode moderne avec :
- `flwr run . local-simulation`

a également été exécuté avec succès sur 3 rounds, avec :
- `aggregate_fit: received 3 results and 0 failures`
- `aggregate_evaluate: received 3 results and 0 failures`
- progression cohérente des métriques

## Décision d’acceptation
La V1 locale est considérée comme **acceptée fonctionnellement**.

Elle remplit son objectif principal :
mettre en place un prototype FL distribué minimal, propre, cohérent avec la baseline, et prêt à être dockerisé puis déployé dans un environnement plus réaliste.

## Limites connues
Les limites suivantes sont connues et acceptées à ce stade :
- le launcher Flower historique émet des warnings de dépréciation
- la simulation Flower moderne sous Windows génère du bruit runtime lié à Ray
- le support Ray sous Windows reste moins stable qu’un environnement Linux/WSL2
- l’évaluation locale utilise encore le dataset local traité, sans séparation complète train/val locale
- la V1 ne contient pas encore :
  - EdgeX
  - MQTT
  - Grafana
  - MLflow
  - déploiement edge réel

## Conclusion
La V1 locale est suffisamment stable pour servir de :
- preuve de faisabilité
- base d’industrialisation légère
- fondation pour la phase Docker
- point de départ de la V2 orientée déploiement et edge computing