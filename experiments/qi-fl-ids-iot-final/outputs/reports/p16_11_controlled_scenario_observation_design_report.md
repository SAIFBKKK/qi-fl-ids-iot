# P16.11 Controlled Scenario Observation Design Report

## Objective

P16.11 prepares the design for future controlled Kali scenario observations. It defines scenario labels, expected feature tendencies, PacketWindow(30) linkage, MQTT-to-IDS evidence flow, dashboard evidence, and safety boundaries.

No scenario is executed in this phase.

## Validated Dependencies

- P16.9.2 `/demo` dashboard is available.
- P16.10 Kali readiness is complete.
- `lab-attacker-kali` has host-only readiness for future controlled design.
- VM1 and VM2 remain the IoT observation nodes.
- Runtime scaler JSON is available from earlier live lab phases.
- final model remains `p8_fedavg_qga_l1`.
- selected mask remains `conservative_seed_42`.

## Selected Scenarios

1. `icmp_flood_like`
   - Family: DoS/DDoS
   - Scientific reference tool: hping3
   - Expected features: `ICMP`, `Rate`, `IAT`, `Number`

2. `tcp_syn_recon_like`
   - Family: Recon / DoS SYN
   - Scientific reference tools: nmap / hping3
   - Expected features: `TCP`, `syn_flag_number`, `syn_count`, `Rate`

3. `http_slow_like`
   - Family: Web-Based / HTTP Flood
   - Scientific reference tools: slowloris / golang-httpflood
   - Expected features: `HTTP`, `TCP`, `Duration`, `IAT`, `Rate`

Tool names are references only. P16.11 stores no executable command lines for these tools.

## Observation Architecture

Target design:

`Kali scenario label -> observation window -> PacketWindow(30) -> 28-feature schema -> scaler JSON -> MQTT ids/flows/{node_id} -> final-mqtt-bridge -> final-ids-api -> ids/predictions/{node_id} and ids/alerts/{node_id} -> dashboard /demo`

## Feature Expectations

- `icmp_flood_like`: `ICMP` should increase, `Rate` may increase, `IAT` may decrease, and `Number` may increase depending on window density.
- `tcp_syn_recon_like`: `TCP` should be active, `syn_flag_number` should increase, `syn_count` should increase, and `Rate` may increase.
- `http_slow_like`: `HTTP` and `TCP` should be active, `Duration` may increase, `IAT` may change, and `Rate` may remain moderate or vary.

These are live-lab tendencies, not a scientific reproduction of CICIoT2023 generation. Final scientific evaluation remains P12/P13.

## Evidence Template

`observation_evidence_template.md` defines future evidence sections:

- scenario_id,
- timestamp,
- Kali VM state,
- target node,
- observation method,
- PacketWindow summary,
- extracted features,
- scaled mode,
- MQTT flow topic,
- prediction topic,
- alert topic,
- dashboard evidence,
- errors,
- interpretation,
- limits.

## Safety Boundaries

- Design only in P16.11.
- Kali is ready but not activated.
- No scenario execution.
- No scan.
- No flood.
- No live capture.
- No traffic generation.
- No training or Flower.
- No changes to P8-P16.10 scientific results.

## Next Step

Recommended next step: P16.12 controlled observation dry-run labels. That phase should attach scenario labels to controlled synthetic PacketWindow(30) payloads without live scenario activation.
