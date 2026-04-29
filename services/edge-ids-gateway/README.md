

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

## Endpoints disponibles

- `GET /`
- `GET /health`
- `GET /ready`
- `GET /metrics`
- `POST /validate/raw`
- `POST /map/features`
- `POST /infer/raw`

## P7.3 - Raw schema validation

`raw_schema.py` valide et normalise le payload brut `raw_iot_event` avant les
phases futures de mapping et d'inference.

Ce que fait P7.3:

- verification des champs requis
- verification des types
- validation IP / ports / protocole
- normalisation de `protocol`
- completion des `flags`
- completion des `flag_counts`
- deduction ou estimation de:
  - `protocol_number`
  - `app_proto`
  - `ttl`
  - `header_bytes_total`
  - `min_packet_size`
  - `packet_size_std`
  - `iat_ns_mean`
  - `window_packet_mean`
  - `request_rate`

Ce que P7.3 ne fait toujours pas:

- aucune connexion MQTT reelle
- aucun mapping vers les 28 features
- aucune inference modele
- aucun chargement du bundle FL

## P7.4 - Raw event to 28-feature mapping

`feature_mapper.py` convertit un evenement brut valide vers le vecteur de 28
features attendu par le modele IDS CIC-IoT.

Le mapping est deterministe, numerique, et conserve l'ordre canonique suivant:

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

Endpoint ajoute:

- `POST /map/features`

Reponse attendue:

```json
{
  "mapped": true,
  "feature_count": 28,
  "feature_names": ["flow_duration", "Header_Length"],
  "features": {
    "flow_duration": 0.085,
    "Header_Length": 240.0
  },
  "feature_vector": [0.085, 240.0]
}
```

Exemple d'entree:

```json
{
  "schema_version": "1.0",
  "event_type": "raw_iot_event",
  "timestamp": "2026-04-28T12:00:00Z",
  "node_id": "sensor-a1",
  "gateway_id": "node1",
  "node_group": "room-a",
  "device_type": "thermostat",
  "src_ip": "10.10.1.21",
  "dst_ip": "10.10.0.10",
  "src_port": 51544,
  "dst_port": 443,
  "protocol": "tcp",
  "packet_size": 820,
  "packet_count": 6,
  "duration_ms": 85,
  "bytes_in": 920,
  "bytes_out": 3980,
  "flags": {
    "syn": 1
  },
  "flag_counts": {},
  "scenario": "normal_traffic"
}
```

Limites scientifiques explicites:

- `Duration` est mappe vers `ttl`
- `IAT` est estime si absent
- `Number` est mappe vers `window_packet_mean`
- `Header_Length`, `Std`, `Min` dependent de champs simules ou estimes

Cette phase ne fait toujours ni MQTT, ni inference, ni chargement du bundle.

## P7.5 - Local inference with deployment bundle

P7.5 ajoute un pipeline local complet dans la gateway:

- `validate_raw_event`
- `map_raw_to_features`
- mise en ordre des 28 features
- `feature_names.pkl`
- `scaler.pkl`
- `global_model.pth`
- softmax
- prediction locale `allow` / `block`

### Composants ajoutes

- `preprocessor.py`
  - charge `feature_names.pkl`
  - charge `scaler.pkl`
  - verifie que l'ordre du bundle correspond a `CANONICAL_FEATURE_NAMES`
  - applique `scaler.transform`

- `inference_api.py`
  - charge `model_config.json`
  - reconstruit la MLP `28 -> 256 -> 128 -> 34`
  - charge `global_model.pth`
  - charge `label_mapping.json`
  - applique softmax
  - retourne label, confiance, alerte, severite

### Fichiers du bundle requis

Dans `experiments/fl-iot-ids-v3/outputs/deployment/baseline_fedavg_normal_classweights/`:

- `global_model.pth`
- `scaler.pkl`
- `feature_names.pkl`
- `label_mapping.json`
- `model_config.json`

### Endpoint d'inference locale

- `POST /infer/raw`

Exemple de payload:

```json
{
  "schema_version": "1.0",
  "event_type": "raw_iot_event",
  "timestamp": "2026-04-28T12:00:00Z",
  "node_id": "sensor-a1",
  "gateway_id": "node1",
  "node_group": "room-a",
  "device_type": "thermostat",
  "src_ip": "10.10.1.21",
  "dst_ip": "10.10.0.10",
  "src_port": 51544,
  "dst_port": 443,
  "protocol": "tcp",
  "packet_size": 820,
  "packet_count": 6,
  "duration_ms": 85,
  "bytes_in": 920,
  "bytes_out": 3980,
  "flags": {
    "syn": 1
  },
  "flag_counts": {},
  "scenario": "normal_traffic"
}
```

Exemple de reponse:

```json
{
  "inferred": true,
  "gateway_id": "node1",
  "node_group": "room-a",
  "flow": {
    "node_id": "sensor-a1",
    "scenario": "normal_traffic",
    "timestamp": "2026-04-28T12:00:00Z"
  },
  "prediction": {
    "predicted_label_id": 1,
    "predicted_label": "BenignTraffic",
    "confidence": 0.987654,
    "is_alert": false,
    "severity": "none",
    "threshold": 0.5
  },
  "decision": "allow",
  "feature_count": 28,
  "latency_ms": 2.34
}
```

### Validation locale

Depuis PowerShell:

```powershell
$bundle="C:\Users\saifb\dev\qi-fl-ids-iot\experiments\fl-iot-ids-v3\outputs\deployment\baseline_fedavg_normal_classweights"
$env:MODEL_PATH="$bundle\global_model.pth"
$env:SCALER_PATH="$bundle\scaler.pkl"
$env:FEATURE_NAMES_PATH="$bundle\feature_names.pkl"
$env:LABEL_MAPPING_PATH="$bundle\label_mapping.json"
cd services\edge-ids-gateway
python -m uvicorn main:app --host 0.0.0.0 --port 8030
```

### Limites P7.5

- inference locale uniquement
- pas encore de MQTT reel
- pas encore d'integration Compose
- la qualite scientifique depend du realisme des features simulees

## P7.6 - MQTT allow/block/predictions/alerts

P7.6 ajoute un vrai collector MQTT a la gateway edge.

Quand `MQTT_ENABLED=true`, la gateway:

- se connecte au broker MQTT
- s'abonne a `RAW_INPUT_TOPIC`
- execute le meme pipeline que `POST /infer/raw`
- publie prediction / accepted / blocked / alert
- publie un statut retained sur `STATUS_TOPIC`

### Variables d'environnement MQTT

| Variable | Defaut |
|---|---|
| `MQTT_ENABLED` | `false` |
| `MQTT_CLIENT_ID` | `edge-ids-gateway-node1` |
| `MQTT_KEEPALIVE` | `30` |
| `MQTT_QOS` | `1` |
| `MQTT_BROKER` | `mosquitto` |
| `MQTT_PORT` | `1883` |
| `MQTT_USERNAME` | `ids_user` |
| `MQTT_PASSWORD` | vide |

### Topics utilises

- input brut: `iot/raw/node1`
- accepted: `iot/accepted/node1`
- blocked: `iot/blocked/node1`
- predictions: `ids/predictions/node1`
- alerts: `ids/alerts/node1`
- status gateway: `ids/status/gateway/node1`

Important:

- Mode A n'est pas modifie
- `ids/status/node1` n'est pas utilise par la gateway
- pas encore d'integration Docker Compose dans cette phase
- pas encore de flows Node-RED geres par le repository

### Messages publies

Prediction:

```json
{
  "schema_version": "1.0",
  "event_type": "edge_ids_prediction",
  "gateway_id": "node1",
  "node_group": "room-a",
  "source_node_id": "sensor-a1",
  "scenario": "normal_traffic",
  "timestamp": "2026-04-28T12:00:00Z",
  "prediction": {
    "predicted_label_id": 18,
    "predicted_label": "DoS-HTTP_Flood",
    "confidence": 0.809248,
    "is_alert": true,
    "severity": "medium",
    "threshold": 0.5,
    "model_version": "baseline_fedavg_normal_classweights"
  },
  "decision": "block",
  "latency_ms": 4.358
}
```

Blocked:

```json
{
  "schema_version": "1.0",
  "event_type": "edge_ids_blocked",
  "gateway_id": "node1",
  "node_group": "room-a",
  "decision": "block",
  "reason": "ids_alert"
}
```

Alert:

```json
{
  "schema_version": "1.0",
  "event_type": "edge_ids_alert",
  "gateway_id": "node1",
  "node_group": "room-a",
  "source_node_id": "sensor-a1",
  "scenario": "normal_traffic",
  "timestamp": "2026-04-28T12:00:00Z",
  "alert": {
    "label": "DoS-HTTP_Flood",
    "confidence": 0.809248,
    "severity": "medium",
    "decision": "block"
  }
}
```

Status retained:

```json
{
  "schema_version": "1.0",
  "event_type": "edge_gateway_status",
  "service": "edge-ids-gateway",
  "gateway_id": "node1",
  "node_group": "room-a",
  "status": "online",
  "mqtt_connected": true,
  "model_ready": true,
  "timestamp": "2026-04-28T12:00:00Z"
}
```

### Tests Mosquitto

```powershell
cd services
docker compose up -d mosquitto
cd ..
```

Puis lancer la gateway avec `MQTT_ENABLED=true`, s'abonner a `ids/#` et `iot/#`,
et publier un evenement sur `iot/raw/node1`.

## P7.7 - Health, readiness and metrics hardening

P7.7 durcit l'observabilite de la gateway avant l'integration Docker Compose.

### Difference entre `/health` et `/ready`

- `GET /health`
  - endpoint de liveness
  - repond `status=ok` tant que le process FastAPI tourne
  - ne tombe pas simplement parce que MQTT est deconnecte

- `GET /ready`
  - endpoint de readiness
  - exige que le bundle inference soit pret
  - si `MQTT_ENABLED=false`, MQTT ne bloque pas la readiness
  - si `MQTT_ENABLED=true`, la connexion MQTT devient requise

### Endpoint de diagnostic

- `GET /diagnostics`

Ce endpoint retourne:

- l'etat runtime agrege
- les topics configures
- l'etat MQTT
- l'etat inference
- les chemins d'artefacts et leur existence

Secrets masques:

- aucun mot de passe MQTT n'est retourne
- seulement `mqtt_username_set` et `mqtt_password_set`

### Metriques Prometheus

En plus des compteurs existants, la gateway expose maintenant clairement:

- `edge_gateway_status`
- `edge_gateway_ready`
- `edge_gateway_inference_ready`
- `edge_gateway_mqtt_connected`
- `edge_gateway_model_ready`
- `edge_gateway_artifact_missing_total`
- `edge_gateway_mqtt_messages_total`

### Exemples

`GET /health`

```json
{
  "status": "ok",
  "service": "edge-ids-gateway",
  "gateway_id": "node1",
  "node_group": "room-a",
  "version": "p7.7-observability-hardening",
  "mode": "local_inference",
  "mqtt_enabled": true,
  "mqtt_connected": true,
  "inference_ready": true
}
```

`GET /ready`

```json
{
  "service": "edge-ids-gateway",
  "version": "p7.7-observability-hardening",
  "gateway_id": "node1",
  "node_group": "room-a",
  "mode": "local_inference",
  "mqtt_enabled": true,
  "mqtt_configured": true,
  "mqtt_connected": true,
  "model_configured": true,
  "model_ready": true,
  "scaler_ready": true,
  "feature_names_ready": true,
  "inference_ready": true,
  "ready": true,
  "reason": null
}
```

`GET /diagnostics`

```json
{
  "service": "edge-ids-gateway",
  "version": "p7.7-observability-hardening",
  "ready": true,
  "mqtt": {
    "enabled": true,
    "configured": true,
    "connected": true,
    "mqtt_username_set": true,
    "mqtt_password_set": true
  },
  "artifacts": {
    "model_path_exists": true,
    "scaler_path_exists": true,
    "feature_names_path_exists": true,
    "label_mapping_path_exists": true
  }
}
```

### Cas MQTT desactive

Si `MQTT_ENABLED=false`:

- `/health` repond `ok`
- `/ready` peut etre `true` si le bundle inference est pret
- `mqtt_connected=false` est normal

### Cas MQTT active

Si `MQTT_ENABLED=true`:

- `/ready` exige `mqtt_connected=true`
- raison d'indisponibilite attendue si broker indisponible:
  - `mqtt_enabled_but_not_connected`

### Limites restantes

- pas encore d'integration Docker Compose pour la gateway
- pas encore de flows Node-RED geres dans le repo
- la qualite scientifique depend toujours du realisme des features simulees
- prochaine etape logique: profile Compose gateway

## P7.8 - Docker Compose profile gateway

P7.8 ajoute l'integration Docker Compose optionnelle via le profile `gateway`.

Depuis `services/`:

```bash
docker compose --profile gateway up -d --build edge-ids-gateway
```

Le service Compose:

- expose l'API sur <http://localhost:8030>
- active `MQTT_ENABLED=true`
- se connecte a `mosquitto:1883`
- s'abonne a `iot/raw/node1`
- publie sur `iot/accepted/node1`, `iot/blocked/node1`,
  `ids/predictions/node1`, `ids/alerts/node1`,
  `ids/status/gateway/node1`
- monte le bundle FL en lecture seule dans `/artifacts`

Mode A reste inchange: `docker compose up -d` ne lance pas
`edge-ids-gateway`.

Prometheus inclut un job `edge-ids-gateway`; le target peut etre `DOWN` quand le
profile `gateway` n'est pas actif.

### Endpoint de validation

`POST /validate/raw`

Reponse OK:

```json
{
  "valid": true,
  "event": {
    "schema_version": "1.0",
    "event_type": "raw_iot_event"
  }
}
```

Reponse KO:

```json
{
  "valid": false,
  "error": "field 'src_ip' must be a valid IPv4 or IPv6 address"
}
```

### Exemple JSON valide

```json
{
  "schema_version": "1.0",
  "event_type": "raw_iot_event",
  "timestamp": "2026-04-28T12:00:00Z",
  "node_id": "sensor-a1",
  "gateway_id": "node1",
  "node_group": "room-a",
  "device_type": "thermostat",
  "src_ip": "10.10.1.21",
  "dst_ip": "10.10.0.10",
  "src_port": 51544,
  "dst_port": 443,
  "protocol": "tcp",
  "packet_size": 820,
  "packet_count": 6,
  "duration_ms": 85,
  "bytes_in": 920,
  "bytes_out": 3980,
  "flags": {
    "syn": 1
  },
  "flag_counts": {},
  "scenario": "normal_traffic"
}
```

### Exemple JSON invalide

```json
{
  "schema_version": "1.0",
  "event_type": "raw_iot_event",
  "timestamp": "2026-04-28T12:00:00Z",
  "node_id": "sensor-a1",
  "gateway_id": "node1",
  "node_group": "room-a",
  "device_type": "thermostat",
  "src_ip": "bad-ip",
  "dst_ip": "10.10.0.10",
  "src_port": 51544,
  "dst_port": 443,
  "protocol": "tcp",
  "packet_size": 820,
  "packet_count": 6,
  "duration_ms": 85,
  "bytes_in": 920,
  "bytes_out": 3980,
  "flags": {
    "syn": 1
  },
  "flag_counts": {},
  "scenario": "normal_traffic"
}
```

## Variables d'environnement

| Variable | Defaut |
|---|---|
| `GATEWAY_ID` | `node1` |
| `NODE_GROUP` | `room-a` |
| `MQTT_ENABLED` | `false` |
| `MQTT_CLIENT_ID` | `edge-ids-gateway-node1` |
| `MQTT_KEEPALIVE` | `30` |
| `MQTT_QOS` | `1` |
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
| `MODEL_CONFIG_PATH` | vide, deduit depuis le dossier du bundle |
| `INFERENCE_THRESHOLD` | `0.5` |

## Validation locale

Depuis la racine du repository:

```powershell
python -m py_compile (Get-ChildItem services/edge-ids-gateway/*.py | ForEach-Object { $_.FullName })
python -c "from pathlib import Path; import sys; sys.path.insert(0, str(Path('services/edge-ids-gateway').resolve())); from raw_schema import validate_raw_event; print(validate_raw_event({'schema_version':'1.0','event_type':'raw_iot_event','timestamp':'2026-04-28T12:00:00Z','node_id':'sensor-a1','gateway_id':'node1','node_group':'room-a','device_type':'thermostat','src_ip':'10.10.1.21','dst_ip':'10.10.0.10','src_port':51544,'dst_port':443,'protocol':'tcp','packet_size':820,'packet_count':6,'duration_ms':85,'bytes_in':920,'bytes_out':3980,'flags':{'syn':1},'flag_counts':{},'scenario':'normal_traffic'})['protocol_number'])"
python -c "from pathlib import Path; import sys; sys.path.insert(0, str(Path('services/edge-ids-gateway').resolve())); from preprocessor import EdgeFeaturePreprocessor; bundle=Path('experiments/fl-iot-ids-v3/outputs/deployment/baseline_fedavg_normal_classweights'); p=EdgeFeaturePreprocessor(str(bundle/'feature_names.pkl'), str(bundle/'scaler.pkl')); print(len(p.feature_names))"
python -c "from pathlib import Path; import sys; sys.path.insert(0, str(Path('services/edge-ids-gateway').resolve())); from inference_api import EdgeInferenceEngine; bundle=Path('experiments/fl-iot-ids-v3/outputs/deployment/baseline_fedavg_normal_classweights'); e=EdgeInferenceEngine(str(bundle/'global_model.pth'), str(bundle/'label_mapping.json'), str(bundle/'model_config.json'), threshold=0.5); print(e.input_dim, e.num_classes, len(e.id_to_label))"
python -c "from pathlib import Path; import sys; sys.path.insert(0, str(Path('services/edge-ids-gateway').resolve())); from collector import MQTTEdgeGatewayCollector; print(MQTTEdgeGatewayCollector.__name__)"
cd services/edge-ids-gateway
python -m uvicorn main:app --host 0.0.0.0 --port 8030
curl http://localhost:8030/
curl http://localhost:8030/health
curl http://localhost:8030/ready
curl http://localhost:8030/diagnostics
curl -X POST http://localhost:8030/validate/raw -H "Content-Type: application/json" -d "{\"schema_version\":\"1.0\",\"event_type\":\"raw_iot_event\",\"timestamp\":\"2026-04-28T12:00:00Z\",\"node_id\":\"sensor-a1\",\"gateway_id\":\"node1\",\"node_group\":\"room-a\",\"device_type\":\"thermostat\",\"src_ip\":\"10.10.1.21\",\"dst_ip\":\"10.10.0.10\",\"src_port\":51544,\"dst_port\":443,\"protocol\":\"tcp\",\"packet_size\":820,\"packet_count\":6,\"duration_ms\":85,\"bytes_in\":920,\"bytes_out\":3980,\"flags\":{\"syn\":1},\"flag_counts\":{},\"scenario\":\"normal_traffic\"}"
curl -X POST http://localhost:8030/map/features -H "Content-Type: application/json" -d "{\"schema_version\":\"1.0\",\"event_type\":\"raw_iot_event\",\"timestamp\":\"2026-04-28T12:00:00Z\",\"node_id\":\"sensor-a1\",\"gateway_id\":\"node1\",\"node_group\":\"room-a\",\"device_type\":\"thermostat\",\"src_ip\":\"10.10.1.21\",\"dst_ip\":\"10.10.0.10\",\"src_port\":51544,\"dst_port\":443,\"protocol\":\"tcp\",\"packet_size\":820,\"packet_count\":6,\"duration_ms\":85,\"bytes_in\":920,\"bytes_out\":3980,\"flags\":{\"syn\":1},\"flag_counts\":{},\"scenario\":\"normal_traffic\"}"
curl -X POST http://localhost:8030/infer/raw -H "Content-Type: application/json" -d "{\"schema_version\":\"1.0\",\"event_type\":\"raw_iot_event\",\"timestamp\":\"2026-04-28T12:00:00Z\",\"node_id\":\"sensor-a1\",\"gateway_id\":\"node1\",\"node_group\":\"room-a\",\"device_type\":\"thermostat\",\"src_ip\":\"10.10.1.21\",\"dst_ip\":\"10.10.0.10\",\"src_port\":51544,\"dst_port\":443,\"protocol\":\"tcp\",\"packet_size\":820,\"packet_count\":6,\"duration_ms\":85,\"bytes_in\":920,\"bytes_out\":3980,\"flags\":{\"syn\":1},\"flag_counts\":{},\"scenario\":\"normal_traffic\"}"
curl http://localhost:8030/metrics
```

## Roadmap P7

- `P7.4` feature mapper 28 features
- `P7.5` inference locale avec bundle
- `P7.6` MQTT allow/block/predictions/alerts
- `P7.7` health/readiness/metrics completes
- `P7.8` integration Docker Compose profile gateway
