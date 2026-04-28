# edge-ids-gateway

Service optionnel prepare pour la future gateway IDS edge du projet QI-FL-IDS-IoT.

## Etat actuel

P7.2 fournit uniquement un squelette Python/FastAPI bootable:

- aucune connexion MQTT reelle
- aucune inference modele
- aucun mapping brut -> 28 features
- aucune integration Docker Compose

Le Mode A existant n'est pas modifie.

## Role cible

A terme, ce service devra:

- recevoir des evenements IoT bruts depuis Node-RED via MQTT
- valider le schema brut
- mapper le payload vers les 28 features CIC-IoT attendues par le bundle
- appliquer `scaler.pkl` + `global_model.pth`
- decider `allow` / `block` / `alert`
- publier accepted / blocked / predictions / alerts / status
- exposer health, readiness et metrics

## Endpoints disponibles en P7.2

- `GET /`
- `GET /health`
- `GET /ready`
- `GET /metrics`

## Variables d'environnement

| Variable | Defaut |
|---|---|
| `GATEWAY_ID` | `node1` |
| `NODE_GROUP` | `room-a` |
| `MQTT_BROKER` | `mosquitto` |
| `MQTT_PORT` | `1883` |
| `MQTT_USERNAME` | `ids_user` |
| `MQTT_PASSWORD` | vide |
| `RAW_INPUT_TOPIC` | `iot/raw/node1` |
| `ACCEPTED_TOPIC` | `iot/accepted/node1` |
| `BLOCKED_TOPIC` | `iot/blocked/node1` |
| `PREDICTIONS_TOPIC` | `ids/predictions/node1` |
| `ALERTS_TOPIC` | `ids/alerts/node1` |
| `STATUS_TOPIC` | `ids/status/gateway/node1` |
| `MODEL_PATH` | `/artifacts/global_model.pth` |
| `SCALER_PATH` | `/artifacts/scaler.pkl` |
| `FEATURE_NAMES_PATH` | `/artifacts/feature_names.pkl` |
| `LABEL_MAPPING_PATH` | `/artifacts/label_mapping.json` |
| `INFERENCE_THRESHOLD` | `0.5` |

## Validation locale

Depuis la racine du repository:

```powershell
python -m py_compile services/edge-ids-gateway/*.py
docker build -t qi-edge-ids-gateway:p7.2 services/edge-ids-gateway
docker run --rm -d --name edge-ids-gateway-p7-2 -p 8030:8000 qi-edge-ids-gateway:p7.2
curl http://localhost:8030/
curl http://localhost:8030/health
curl http://localhost:8030/ready
curl http://localhost:8030/metrics
docker stop edge-ids-gateway-p7-2
```

## Roadmap P7

- `P7.3` raw schema validator
- `P7.4` feature mapper 28 features
- `P7.5` inference locale avec bundle
- `P7.6` MQTT allow/block/predictions/alerts
- `P7.7` health/readiness/metrics completes
- `P7.8` integration Docker Compose profile gateway
