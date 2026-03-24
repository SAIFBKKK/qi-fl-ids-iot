# Release Note — v1.1-docker-config-stable

## Objet
Ce document fige le premier jalon reproductible du système FL pour IDS IoT.

## Statut
Cette version est considérée comme stable et reproductible dans le runtime canonique Docker Compose.

## Périmètre validé
- baseline centralisée conservée comme source de vérité
- données locales figées
- preprocessing local validé
- entraînement mono-client validé
- exécution FL classique validée
- image Docker CPU-only validée
- orchestration Docker Compose validée
- configuration centralisée YAML validée

## Résultat système validé
Scénario validé :
- 1 serveur Flower
- 3 clients Flower
- 3 rounds
- agrégation fit sans échec
- agrégation evaluate sans échec
- arrêt propre des conteneurs

## Éléments figés dans ce jalon
- `src/`
- `configs/fl_config.yaml`
- `deployments/docker/Dockerfile`
- `deployments/docker/docker-compose.yml`
- `artifacts/`
- `requirements.txt`
- `requirements-lock.txt`
- `pyproject.toml`

## Politique de données
Les datasets volumineux ne sont pas nécessairement versionnés directement dans Git.
Leur état doit rester figé localement à ce stade.
La future étape MLOps pourra introduire DVC ou Git LFS si nécessaire.

## Hors périmètre
Cette release ne contient pas encore :
- MLflow
- DVC
- EdgeX
- MQTT
- Grafana
- SuperLink / SuperNode Flower moderne complet
- modules quantum-inspired
- expérimentation comparative QGA / QGWO

## Rôle de ce jalon
Cette version sert de :
- baseline FL reproductible
- point de retour stable
- référence de comparaison pour les prochaines évolutions

## Nom du tag Git recommandé
`v1.1-docker-config-stable`