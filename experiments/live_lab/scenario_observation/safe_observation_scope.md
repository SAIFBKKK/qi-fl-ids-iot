# P16.11 Safe Observation Scope

P16.11 is a design-only phase.

## Active Boundaries

- No scenario is executed in P16.11.
- Kali is ready but not activated.
- No scan is run.
- No flood is run.
- No live packet capture is started.
- No traffic generation is performed.
- No training or Flower process is launched.
- No scientific results from P8-P16.10 are modified.

## Isolation

The lab remains on the VirtualBox host-only network:

- Server: `192.168.56.1`
- VM1: `192.168.56.101`
- VM2: `192.168.56.102`
- VM3: `192.168.56.103`

Future scenario activation must be manually approved, isolated, and documented before it happens.

## Current Demonstration Baseline

The current live demo remains based on controlled PacketWindow(30) feature publication:

`PacketWindow(30) -> scaler JSON -> MQTT -> final-mqtt-bridge -> final-ids-api -> dashboard /demo`
