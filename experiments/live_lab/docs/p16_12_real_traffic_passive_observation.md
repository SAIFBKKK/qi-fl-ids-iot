# P16.12 Real-Traffic Passive Observation

## Objective

P16.12 prepares VM1 for passive real-traffic observation:

`passive capture -> PacketWindow(30) -> 28 features -> scaler JSON -> MQTT -> IDS -> dashboard`

The agent observes local packets only. It does not generate traffic, does not start a scenario, and does not store application payloads.

## Architecture

- Server Windows: `192.168.56.1`
- VM1 `iot-drone-sitl`: `192.168.56.101`
- VM3 `lab-attacker-kali`: `192.168.56.103`
- VM1 interface: `enp0s8`
- MQTT broker: `192.168.56.1:1883`
- dashboard demo: `http://192.168.56.1:8013/demo`

## Passive Observation Path

1. VM1 observes packets on `enp0s8`.
2. The capture filter is limited to the host-only network and VM1 IP.
3. Packets are normalized without application payload storage.
4. `PacketWindow(30)` groups observed metadata.
5. `extract_28_features_from_window` builds the 28-feature vector.
6. The runtime scaler JSON prepares `selected_12_scaled` or `original_28_scaled`.
7. MQTT publishes to `ids/flows/iot-drone-sitl`.
8. final-mqtt-bridge calls final-ids-api.
9. Predictions and alerts appear on `/demo`.

## Safe Benign Test

Use only normal lab activity that already happens during the demo, such as service health checks, dashboard refreshes, or VM registration. P16.12 does not provide any traffic-generation command.

## VM1 Dry Run

```bash
cd ~/qi-fl-ids-iot
git pull
python3 experiments/qi-fl-ids-iot-final/src/scripts/16_12_run_passive_observation_publish.py --broker 192.168.56.1 --node-id iot-drone-sitl --interface enp0s8 --node-ip 192.168.56.101 --peer-ip 192.168.56.103 --input-mode selected_12_scaled --window-size 30 --max-windows 1 --timeout-seconds 30 --dry-run --allow-live-capture
```

## VM1 Publish After Dry Run

```bash
python3 experiments/qi-fl-ids-iot-final/src/scripts/16_12_run_passive_observation_publish.py --broker 192.168.56.1 --node-id iot-drone-sitl --interface enp0s8 --node-ip 192.168.56.101 --peer-ip 192.168.56.103 --input-mode selected_12_scaled --window-size 30 --max-windows 1 --timeout-seconds 30 --publish --allow-live-capture
```

## Limits

- Feature extraction is approximate.
- It is not a complete CICIoT2023 extractor.
- Application payloads are not stored.
- No scenario is launched by this repository.
- The live capture mode requires explicit `--allow-live-capture`.
- Scientific evaluation remains P12/P13.

