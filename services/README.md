# Microservices stack - QI-FL-IDS-IoT

Architecture microservices pour le framework QI-FL-IDS-IoT.
9 services always-on + 2 services profiles.

## Quick Start

```bash
cd services

# 1. Configurer
cp .env.example .env
# Editer .env, au minimum changer MQTT_PASSWORD et GRAFANA_ADMIN_PASSWORD.

# 2. Generer le password MQTT
./scripts/generate_mqtt_password.sh

# 3. Construire les subsets de demo
../.venv/Scripts/python.exe scripts/build_demo_subsets.py

# 4. Demarrer la stack infra P1
docker-compose --env-file .env up -d

# 5. Verifier
./scripts/healthcheck_all.sh
```

> **Note sur les chemins de donnees :**
> Le dataset complet CIC-IoT-2023 est dans
> `data/balancing_v3_fixed300k_outputs/` (gitignored).
> Les subsets de demo generes par `build_demo_subsets.py` sont dans
> `data/cic-iot-2023/demo_subsets/` (versionnes Git, 4.7 MB generes en P1).
> Le bundle FL baseline est dans
> `experiments/fl-iot-ids-v3/outputs/deployment/baseline_fedavg_normal_classweights/`
> (versionne Git, environ 200 KB).

## Services

| Service | Port | Profile | Role |
|---|---:|---|---|
| traffic-generator | 8010 | default | Replay CIC-IoT-2023 |
| iot-node-1/2/3 | 8001/2/3 | default | Inference + FL client |
| fl-server | 8080 | training | FL orchestrator |
| mosquitto | 1883 | default | MQTT broker |
| mlflow | 5000 | default | Experiment tracking |
| prometheus | 9090 | default | Metrics scraping |
| grafana | 3000 | default | Dashboards |
| node-red | 1880 | orchestration | Scenario orchestration |
| qga-service | - | preprocessing | QGA feature selection |

## Profiles

```bash
# Demo de base, Mode A
docker-compose --env-file .env up -d

# Demo avec scenarios Node-RED, Mode C
docker-compose --env-file .env --profile orchestration up -d

# Round FL training
docker-compose --env-file .env --profile training up fl-server

# QGA preprocessing
docker-compose --env-file .env --profile preprocessing run --rm qga-service
```

## Structure

Voir `services/<service>/README.md` pour chaque service individuel.

## Scripts Utilitaires

- `scripts/build_demo_subsets.py` - Genere les 6 fichiers parquet de demo
- `scripts/generate_mqtt_password.sh` - Cree le password file Mosquitto
- `scripts/reset.sh` - Reset propre soft ou complet avec `--hard`
- `scripts/healthcheck_all.sh` - Verifie l'etat des services infra P1
- `scripts/test_publish.py` - Publie des flows de test MQTT depuis les demo subsets

## Implementation Status

- [x] P1 - Foundation (squelette + scripts)
- [x] P2 - iot-node service
- [x] P3 - traffic-generator
- [ ] P4 - Compose Mode A complet
- [ ] P5 - fl-server
- [ ] P6 - Monitoring dashboards Grafana + qga-service
- [ ] P7 - Node-RED scenarios
- [ ] P8 - Tests E2E
- [ ] P9 - Documentation finale

## P2 Validation Snapshot

P2 a ete valide sur `iot-node-1` avec le bundle US1 reel et MQTT authentifie.

- Health: `/health` retourne `status=ok`, `mqtt_connected=true`, `inference_engine=torch_mlp`
- Pipeline MQTT: 5 flows `ddos_burst` recus, 5 predictions publiees, 5 alerts publiees
- Schema strict: 0 rejet, aucune imputation silencieuse
- Labels observes: `DDoS-UDP_Flood`, `DDoS-RSTFINFlood`, `DDoS-SlowLoris`, `DDoS-SYN_Flood`
- Latence CPU edge: environ 2 ms par flow, soit moins de 5 ms par inference
- Metrics Prometheus: `ids_flows_received_total`, `ids_flows_rejected_invalid_schema_total`, `ids_predictions_total`, `ids_alerts_total`, `inference_latency_seconds`, `ids_node_status`

## P3 Validation Snapshot

P3 a ete valide avec la chaine complete `traffic-generator -> MQTT -> iot-node-1 -> modele PyTorch -> predictions/alerts -> Prometheus`.

- Flux recus par `iot-node-1`: `ids_flows_received_total=16778`
- Predictions: `ids_predictions_total` augmente sur plusieurs classes CIC-IoT-2023
- Alerts: `ids_alerts_total` actif avec severites `low`, `medium`, `high`, `critical`
- Rejets schema/features: `0` sur toutes les raisons exposees
- Latence inference: `7.626942627s` pour `16778` flows, soit environ `0.45 ms/flow`
- Node status: `ids_node_status{node_id="node1"} 1`
