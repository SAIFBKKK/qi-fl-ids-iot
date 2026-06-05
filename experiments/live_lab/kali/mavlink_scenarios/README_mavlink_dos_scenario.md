# MAVLink Controlled DoS-Like Scenario Notes

QI-FL-IDS-IoT project author: Saif Ben Fredj.
Phase 2 Live Lab - 2026-06-05.
Description: scientific and operational notes for a future supervised MAVLink/UDP high-rate observation scenario.

## Scientific Context

The live lab uses CICIoT2023 as conceptual guidance for observing packet-window behavior and feature tendencies. The drone node is represented as `iot-drone-sitl`, a simulated UAV using MAVLink over UDP. The purpose of this note is to describe what the IDS pipeline should observe when UDP packet density changes.

This file is documentation only. It does not contain executable attack commands, traffic-generation commands, or scanner usage.

## Scenario Architecture

- Server: `192.168.56.1`
- Drone VM: `iot-drone-sitl`, `192.168.56.101`
- Kali workstation: `lab-attacker-kali`, `192.168.56.103`
- Network: VirtualBox Host-Only only
- Observation path: passive or synthetic packet metadata -> `PacketWindow(30)` -> 28 CICIoT2023-like features -> scaler JSON -> QGA mask -> MQTT -> IDS -> dashboard

## Expected Network Features

| Condition | Rate | IAT | UDP | Number | Tot sum | Std |
|---|---:|---:|---:|---:|---:|---:|
| Nominal MAVLink-like telemetry | low, around one packet per second in the safe simulation | around one second | 30 in a full window | 30 | stable | low |
| High-density synthetic window | high, around hundreds of packet metadata items per second | very low | 30 in a full window | 30 | higher or denser | can increase |

TCP features such as `syn_flag_number` and `syn_count` are not expected to increase because MAVLink is represented here as UDP metadata.

## Future Activation Placeholder

Future scenario activation must be supervised and explicitly approved. The current repository-supported approach is the safe `burst-sim` path in `mavlink_passive_agent.py`, which creates packet metadata in memory and does not generate network traffic.

## Scientific Limits

- The live-lab features are safe CICIoT2023-like approximations, not a full reproduction of the original CICIoT2023 generation process.
- The passive agent does not store application payloads.
- The current demo validates deployment behavior. Scientific performance evaluation remains tied to the earlier P12/P13 evidence.
