# fl-server

> Status: PLACEHOLDER - Sera implemente dans le prompt P5

## Role cible

Orchestrer les rounds FL, exposer les metriques serveur et produire les artefacts entraines.

## Variables d'env cible

- `FL_SERVER_HOST`
- `FL_SERVER_PORT`
- `FL_NUM_ROUNDS`
- `MODEL_PATH`

## Endpoints cible

- `GET /health`
- `GET /metrics`
- API FL a definir dans P5

## TODO

- [ ] Implementer Dockerfile
- [ ] Implementer requirements.txt
- [ ] Implementer logique metier
- [ ] Tests unitaires
- [ ] Documenter usage
