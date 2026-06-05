# iot-drone-sitl

QI-FL-IDS-IoT Phase 2 Live Lab - simulated UAV SITL node.

This node replaces the previous VM1 `iot-rpi-weak` role for the live lab while keeping the same host-only IP, `192.168.56.101`.

Role:

- represent an ArduPilot/MAVLink UAV node in a controlled isolated lab;
- observe MAVLink/UDP packet metadata passively;
- build `PacketWindow(30)` windows;
- map windows to the 28 CICIoT2023-like project features;
- apply the runtime scaler JSON and the QGA mask;
- publish `selected_12_scaled` payloads to `ids/flows/iot-drone-sitl`.

Safe dry-run commands:

```bash
python register_drone_node.py --dry-run
python mavlink_passive_agent.py --scenario normal-sim --dry-run
python mavlink_passive_agent.py --scenario burst-sim --dry-run
```

The simulation modes only create packet metadata in memory. They do not generate network traffic.
