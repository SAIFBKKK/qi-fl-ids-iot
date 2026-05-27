# P16.10 Kali Lab Workstation Readiness Plan

## Objective

P16.10 prepares `lab-attacker-kali` as a future controlled scenario workstation without executing scenarios. The phase checks readiness, documents the isolated host-only scope, and records the three CICIoT2023-inspired scenario families selected for future design.

## Architecture

- Server Windows host: `192.168.56.1`
- VM1 `iot-rpi-weak`: `192.168.56.101`
- VM2 `iot-smart-watch-medium`: `192.168.56.102`
- VM3 `lab-attacker-kali`: `192.168.56.103`
- Dashboard demo: `http://192.168.56.1:8013/demo`
- Controller: `http://192.168.56.1:8020`
- final IDS API: `http://192.168.56.1:8014`
- final MQTT bridge: `http://192.168.56.1:8016`
- online-validator: `http://192.168.56.1:8015`
- MQTT broker: `192.168.56.1:1883`

## Dependencies Before Future Activation

- Docker Desktop stack running on the server.
- `/demo` dashboard available.
- VM1 and VM2 registered and visible in live-lab-controller.
- Kali VM on host-only network with `192.168.56.103`.
- NAT disabled during controlled observation windows unless explicitly needed for maintenance.
- Readiness script completed with warnings reviewed.

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

Tool names are scientific references only in P16.10. No executable command line is stored for these tools.

## P16.10 Steps

- P16.10-A: Create `experiments/live_lab/kali/`.
- P16.10-B: Add Kali README and safe scope.
- P16.10-C: Add descriptive scenario catalog.
- P16.10-D: Add safe readiness inventory script.
- P16.10-E: Add VM readiness documentation.
- P16.10-F: Add host-only network isolation documentation.
- P16.10-G: Add selected scenario documentation.
- P16.10-H: Validate structure, safety, and integration tests.

## Limits

- No scenario execution.
- No scan.
- No generated traffic.
- No live capture.
- No training or Flower.
- No changes to P8-P16.9.2 scientific or runtime results.

## Next Step

Next step: P16.11 controlled scenario observation design. P16.11 should define evidence requirements, synthetic or replay boundaries, monitoring views, and go/no-go checks before any future controlled scenario is activated.
