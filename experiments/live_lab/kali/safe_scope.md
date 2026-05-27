# Kali Lab Workstation Safe Scope

`lab-attacker-kali` is prepared as an isolated lab workstation for future controlled CICIoT2023-inspired observations.

## Boundaries

- The Kali VM is part of the host-only live lab network.
- Scientific tool names are references from CICIoT2023 and inventory labels only.
- The repository does not store executable offensive command lines.
- P16.10 does not activate any scenario.
- No scenario is executed in P16.10.
- P16.10 does not run live packet capture.
- No live capture is started in P16.10.
- P16.10 does not generate traffic.
- P16.10 does not scan a network.

## Current Demonstration Path

The current live defense demonstration remains based on controlled PacketWindow(30) feature generation and MQTT publication:

`VM -> PacketWindow(30) -> scaler JSON -> MQTT -> final-mqtt-bridge -> final-ids-api -> predictions/alerts -> dashboard /demo`

## Future Activation Rule

The three selected scenarios may only be designed in a future phase after readiness, isolation, and evidence requirements are reviewed. Any future work must remain inside the isolated lab and must be documented as controlled observation, not unrestricted activity.
