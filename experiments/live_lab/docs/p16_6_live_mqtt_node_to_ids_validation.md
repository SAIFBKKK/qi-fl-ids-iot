# P16.6 Live MQTT Node-to-IDS Validation

P16.6 validates the safe technical path from VM node JSON payloads to MQTT predictions and alerts.

Validated path:

`VM node -> ids/flows/{node_id} -> final-mqtt-bridge -> final-ids-api -> ids/predictions/{node_id} / ids/alerts/{node_id}`

## Observer Commands

From the server or a VM with MQTT tools installed:

```bash
mosquitto_sub -h 192.168.56.1 -p 1883 -u ids_user -P changeme_in_dotenv -t 'ids/flows/#' -v
mosquitto_sub -h 192.168.56.1 -p 1883 -u ids_user -P changeme_in_dotenv -t 'ids/predictions/#' -v
mosquitto_sub -h 192.168.56.1 -p 1883 -u ids_user -P changeme_in_dotenv -t 'ids/alerts/#' -v
```

## VM1 Publication

```bash
cd ~/qi-fl-ids-iot
git pull
python3 experiments/qi-fl-ids-iot-final/src/scripts/16_6_publish_safe_mqtt_payloads.py \
  --broker 192.168.56.1 \
  --node-id iot-rpi-weak \
  --input-mode selected_12_scaled
```

Expected path:

- Publish to `ids/flows/iot-rpi-weak`.
- Receive prediction on `ids/predictions/iot-rpi-weak`.
- Receive alert on `ids/alerts/iot-rpi-weak`.

## VM2 Publication

```bash
cd ~/qi-fl-ids-iot
git pull
python3 experiments/qi-fl-ids-iot-final/src/scripts/16_6_publish_safe_mqtt_payloads.py \
  --broker 192.168.56.1 \
  --node-id iot-smart-watch-medium \
  --input-mode original_28_scaled
```

Expected path:

- Publish to `ids/flows/iot-smart-watch-medium`.
- final-ids-api applies `conservative_seed_42` to the 28 scaled features.
- Receive prediction on `ids/predictions/iot-smart-watch-medium`.
- Receive alert on `ids/alerts/iot-smart-watch-medium`.

## Metrics Commands

```bash
curl http://192.168.56.1:8016/metrics
curl http://192.168.56.1:8015/summary
curl http://192.168.56.1:8014/metrics
curl http://192.168.56.1:8020/nodes
curl http://192.168.56.1:8020/assignments
```

Collect structured evidence:

```bash
python experiments/qi-fl-ids-iot-final/src/scripts/16_6_collect_live_mqtt_evidence.py \
  --server-url http://192.168.56.1
```

## Interpretation

- VM1 validates the 12 selected scaled feature path.
- VM2 validates the 28 original scaled feature path and QGA mask application inside final-ids-api.
- `final_ids_api_prediction_errors_total` should remain `0`.
- `final_mqtt_bridge_prediction_errors_total` should remain `0`.
- One flow should produce one prediction and one alert for each VM node during this technical test.

## Scope

P16.6 uses controlled JSON flow payloads only. It does not validate scientific model performance, does not perform packet capture, and does not start a lab scenario. Scientific performance evidence remains P12/P13.

## Remaining Limit

The live lab still needs P16.7 real-time packet window feature extraction before raw local packets can become 28-feature IDS inputs.
