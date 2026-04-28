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
| `INFERENCE_THRESHOLD` | `0.5` |

## Validation locale

Depuis la racine du repository:

```powershell
python -m py_compile (Get-ChildItem services/edge-ids-gateway/*.py | ForEach-Object { $_.FullName })
python -c "from pathlib import Path; import sys; sys.path.insert(0, str(Path('services/edge-ids-gateway').resolve())); from raw_schema import validate_raw_event; print(validate_raw_event({'schema_version':'1.0','event_type':'raw_iot_event','timestamp':'2026-04-28T12:00:00Z','node_id':'sensor-a1','gateway_id':'node1','node_group':'room-a','device_type':'thermostat','src_ip':'10.10.1.21','dst_ip':'10.10.0.10','src_port':51544,'dst_port':443,'protocol':'tcp','packet_size':820,'packet_count':6,'duration_ms':85,'bytes_in':920,'bytes_out':3980,'flags':{'syn':1},'flag_counts':{},'scenario':'normal_traffic'})['protocol_number'])"
cd services/edge-ids-gateway
python -m uvicorn main:app --host 0.0.0.0 --port 8030
curl http://localhost:8030/
curl http://localhost:8030/health
curl http://localhost:8030/ready
curl -X POST http://localhost:8030/validate/raw -H "Content-Type: application/json" -d "{\"schema_version\":\"1.0\",\"event_type\":\"raw_iot_event\",\"timestamp\":\"2026-04-28T12:00:00Z\",\"node_id\":\"sensor-a1\",\"gateway_id\":\"node1\",\"node_group\":\"room-a\",\"device_type\":\"thermostat\",\"src_ip\":\"10.10.1.21\",\"dst_ip\":\"10.10.0.10\",\"src_port\":51544,\"dst_port\":443,\"protocol\":\"tcp\",\"packet_size\":820,\"packet_count\":6,\"duration_ms\":85,\"bytes_in\":920,\"bytes_out\":3980,\"flags\":{\"syn\":1},\"flag_counts\":{},\"scenario\":\"normal_traffic\"}"
curl -X POST http://localhost:8030/map/features -H "Content-Type: application/json" -d "{\"schema_version\":\"1.0\",\"event_type\":\"raw_iot_event\",\"timestamp\":\"2026-04-28T12:00:00Z\",\"node_id\":\"sensor-a1\",\"gateway_id\":\"node1\",\"node_group\":\"room-a\",\"device_type\":\"thermostat\",\"src_ip\":\"10.10.1.21\",\"dst_ip\":\"10.10.0.10\",\"src_port\":51544,\"dst_port\":443,\"protocol\":\"tcp\",\"packet_size\":820,\"packet_count\":6,\"duration_ms\":85,\"bytes_in\":920,\"bytes_out\":3980,\"flags\":{\"syn\":1},\"flag_counts\":{},\"scenario\":\"normal_traffic\"}"
curl http://localhost:8030/metrics
```

## Roadmap P7

- `P7.4` feature mapper 28 features
- `P7.5` inference locale avec bundle
- `P7.6` MQTT allow/block/predictions/alerts
- `P7.7` health/readiness/metrics completes
- `P7.8` integration Docker Compose profile gateway
