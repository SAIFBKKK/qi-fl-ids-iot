# iot-node

> Status: PLACEHOLDER - Sera implemente dans le prompt P2

## Role cible

Executer la collecte MQTT, le preprocessing, l'inference IDS et le client FL local.

## Variables d'env cible

- `MQTT_BROKER`
- `MODEL_PATH`
- `SCALER_PATH`
- `LABEL_MAPPING_PATH`
- `INFERENCE_THRESHOLD`

## Endpoints cible

- `GET /health`
- `GET /metrics`
- Endpoints inference HTTP a definir dans P2

## TODO

- [ ] Implementer Dockerfile
- [ ] Implementer requirements.txt
- [ ] Implementer logique metier
- [ ] Tests unitaires
- [ ] Documenter usage
