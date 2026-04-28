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
cd services/edge-ids-gateway
python -m uvicorn main:app --host 0.0.0.0 --port 8030
curl http://localhost:8030/
curl http://localhost:8030/health
curl http://localhost:8030/ready
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
