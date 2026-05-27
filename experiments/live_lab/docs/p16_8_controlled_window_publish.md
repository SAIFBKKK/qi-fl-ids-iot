# P16.8 Controlled Packet-Window Publish

P16.8 publishes controlled synthetic packet-window features into the live MQTT-to-IDS pipeline.

## VM1 Dry-Run

```bash
cd ~/qi-fl-ids-iot
git pull
python3 experiments/qi-fl-ids-iot-final/src/scripts/16_8_run_controlled_window_publish.py \
  --broker 192.168.56.1 \
  --node-id iot-rpi-weak \
  --input-mode selected_12_scaled \
  --window-size 30 \
  --max-windows 1 \
  --dry-run
```

## VM1 Publish

```bash
python3 experiments/qi-fl-ids-iot-final/src/scripts/16_8_run_controlled_window_publish.py \
  --broker 192.168.56.1 \
  --node-id iot-rpi-weak \
  --input-mode selected_12_scaled \
  --window-size 30 \
  --max-windows 1 \
  --publish
```

## VM2 Dry-Run

```bash
cd ~/qi-fl-ids-iot
git pull
python3 experiments/qi-fl-ids-iot-final/src/scripts/16_8_run_controlled_window_publish.py \
  --broker 192.168.56.1 \
  --node-id iot-smart-watch-medium \
  --input-mode original_28_scaled \
  --window-size 30 \
  --max-windows 1 \
  --dry-run
```

## VM2 Publish

```bash
python3 experiments/qi-fl-ids-iot-final/src/scripts/16_8_run_controlled_window_publish.py \
  --broker 192.168.56.1 \
  --node-id iot-smart-watch-medium \
  --input-mode original_28_scaled \
  --window-size 30 \
  --max-windows 1 \
  --publish
```

## MQTT Observers

VM1:

```bash
mosquitto_sub -h 192.168.56.1 -p 1883 -u ids_user -P changeme_in_dotenv -t 'ids/flows/iot-rpi-weak' -t 'ids/predictions/iot-rpi-weak' -t 'ids/alerts/iot-rpi-weak' -v
```

VM2:

```bash
mosquitto_sub -h 192.168.56.1 -p 1883 -u ids_user -P changeme_in_dotenv -t 'ids/flows/iot-smart-watch-medium' -t 'ids/predictions/iot-smart-watch-medium' -t 'ids/alerts/iot-smart-watch-medium' -v
```

## Collect Evidence

```bash
python experiments/qi-fl-ids-iot-final/src/scripts/16_8_collect_window_publish_evidence.py \
  --server-url http://192.168.56.1
```

## Interpretation

- VM1 should publish 12 selected scaled features to `ids/flows/iot-rpi-weak`.
- VM2 should publish 28 scaled features to `ids/flows/iot-smart-watch-medium`.
- final-ids-api applies the QGA mask for VM2 inside the API path.
- The MQTT bridge should publish predictions and alerts on the node-specific topics.
- Error counters should remain zero.

## Screenshots Recommended

- Dry-run summary for VM1 and VM2.
- MQTT observer terminal for VM1.
- MQTT observer terminal for VM2.
- final-mqtt-bridge metrics for both nodes.
- online-validator summary showing flow, prediction, and alert topics.
- final-ids-api metrics with error counter at zero.

## Scope

P16.8 uses only synthetic packet windows and controlled MQTT publication. It does not use live capture, pcap files, Kali scenarios, training, or Flower.
