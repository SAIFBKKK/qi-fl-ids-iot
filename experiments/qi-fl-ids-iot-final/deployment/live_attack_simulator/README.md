# P16 Controlled Scenario Simulator

Publishes controlled feature payloads to MQTT for live lab demonstrations. It
does not run network exercises, host probing, credential attempts, spoofing, or
packet generation against machines. The simulator only emits JSON vectors or
replays local JSON/JSONL/CSV feature files into the isolated lab broker.

Allowed scenario labels:

- `benign`
- `ddos_dos_like`
- `recon_like`
- `web_based_like`
- `brute_force_like`
- `spoofing_like`
- `mirai_like`

Example:

```bash
python scenario_publisher.py --node-id iot-drone-sitl --mqtt-host SERVER_IP --scenario recon_like --input-mode original_28_scaled
```

Payload contract:

```json
{
  "flow_id": "sim-iot-drone-sitl-recon_like-original_28_scaled-000001",
  "node_id": "iot-drone-sitl",
  "timestamp": "2026-05-24T12:00:00Z",
  "input_mode": "selected_12_scaled",
  "scenario": "recon_like",
  "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
}
```

For `original_28_scaled`, the `features` array contains 28 scaled values.

