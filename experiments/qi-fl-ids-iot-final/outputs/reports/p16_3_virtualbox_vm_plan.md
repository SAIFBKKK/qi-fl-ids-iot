# P16.3 VirtualBox VM Plan for Live Lab

Date: 2026-05-26
Branch: `final/quantum-inspired-fl-iot-ids-final`

## Final Architecture

The live lab will use one Windows server PC and three VirtualBox VMs on a local
Host-Only network.

Server PC:

- Docker Desktop server stack.
- `SERVER_IP = 192.168.56.1`.
- Exposes MQTT, final IDS API, MQTT bridge, online validator, live lab
  controller, dashboard, Prometheus, and Grafana.

VirtualBox VM storage root:

`E:\VirtualBox VMs\qi-fl-ids-iot-live-lab\`

VMs:

- VM1: `iot-rpi-weak`, Ubuntu Server minimal, raspberry-like weak IoT node.
- VM2: `iot-smart-watch-medium`, Ubuntu Server minimal, smart-watch-like medium
  IoT node.
- VM3: `lab-attacker-kali`, Kali Linux, CICIoT2023-inspired controlled scenario
  workstation.

## Static IP Plan

- Server PC: `192.168.56.1`
- VM1 `iot-rpi-weak`: `192.168.56.101`
- VM2 `iot-smart-watch-medium`: `192.168.56.102`
- VM3 `lab-attacker-kali`: `192.168.56.103`
- Subnet: `255.255.255.0`
- Gateway: none on Host-Only adapter
- Optional second adapter: NAT only during package installation

## Server Ports

- MQTT Mosquitto: `1883`
- dashboard P13: `8013`
- final IDS API: `8014`
- online validator: `8015`
- final MQTT bridge: `8016`
- live lab controller: `8020`
- Prometheus: `9090`
- Grafana: `3000`

## Resource Plan for 16 GB RAM PC

Keep Windows and Docker Desktop as the priority. Use small VM allocations:

| VM | vCPU | RAM | Disk |
|---|---:|---:|---:|
| `iot-rpi-weak` | 1 | 1024 MB | 12 GB |
| `iot-smart-watch-medium` | 1 | 1536 MB | 15 GB |
| `lab-attacker-kali` | 1 | 1536 MB | 20 GB |

Total planned VM RAM: `4096 MB`.

If the PC becomes slow:

- stop Grafana when not collecting dashboard evidence;
- stop one VM not used in the current demo segment;
- reduce `iot-smart-watch-medium` to `1024 MB`;
- keep Kali powered off outside controlled scenario demonstrations.

## VM Roles

### iot-rpi-weak

- Role: raspberry-like weak IoT node.
- OS: Ubuntu Server minimal.
- Mode: server-side inference.
- Publishes controlled feature payloads to `ids/flows/iot-rpi-weak`.
- Receives alerts from `ids/alerts/iot-rpi-weak`.

### iot-smart-watch-medium

- Role: smart-watch-like medium IoT node.
- OS: Ubuntu Server minimal.
- Mode: future edge inference.
- First uses controlled 28-feature replay and server validation.
- Later can compare edge inference against server-side inference.

### lab-attacker-kali

- Role: CICIoT2023-inspired controlled scenario workstation.
- OS: Kali Linux.
- Mode: controlled lab scenarios only.
- Kali is chosen because it is a familiar security lab workstation and a useful
  demonstration environment for explaining CICIoT2023 families. In this repo it
  is not used to run real attacks and no offensive command lines are provided.

## Selected CICIoT2023-Inspired Scenarios

1. `icmp_flood_like`
   - Family: DoS/DDoS.
   - Scientific tool reference: hping3.
   - Expected packet structure: IP + ICMP.
   - Impacted features: `ICMP`, `Rate`, `IAT`, `Number`, `Header_Length`,
     `Tot sum`, `Min`, `Std`.
   - Safe lab usage: controlled replay, local pcap, or feature windows.

2. `tcp_syn_recon_like`
   - Family: Recon / DoS SYN.
   - Scientific tool references: nmap, hping3.
   - Expected packet structure: IP + TCP with SYN-dominant windows.
   - Impacted features: `TCP`, `syn_flag_number`, `syn_count`, `Rate`, `IAT`,
     `Number`, `Header_Length`.
   - Safe lab usage: controlled replay, local pcap, or feature windows.

3. `http_slow_like`
   - Family: Web-Based / HTTP Flood.
   - Scientific tool references: slowloris, golang-httpflood.
   - Expected packet structure: TCP + HTTP requests.
   - Impacted features: `HTTP`, `TCP`, `Duration`, `Rate`, `IAT`,
     `psh_flag_number`, `ack_count`, `Tot sum`.
   - Safe lab usage: controlled replay, local pcap, or feature windows.

## Security Limits

P16.3 does not:

- create VMs automatically;
- start Docker;
- start packet capture;
- run real attack traffic;
- provide executable offensive commands;
- train models;
- run Flower;
- modify P8-P16 results.

The Kali VM is a demonstration and controlled-scenario workstation only.

## Manual VirtualBox Creation Steps

1. Create the folder:
   `E:\VirtualBox VMs\qi-fl-ids-iot-live-lab\`
2. Create VM1 `iot-rpi-weak`.
3. Create VM2 `iot-smart-watch-medium`.
4. Create VM3 `lab-attacker-kali`.
5. Configure Adapter 1 as VirtualBox Host-Only.
6. Optionally configure Adapter 2 as NAT only during installation.
7. Install the selected OS manually.
8. Set each hostname to match the VM name.
9. Configure static IP addresses from this plan.
10. Do not start capture or scenarios during this planning step.

## Manual OS Installation Notes

- Ubuntu Server minimal is enough for the two IoT nodes.
- Kali Linux is reserved for the controlled scenario workstation.
- Keep guest additions and package installation for later setup steps.
- Clone or copy `experiments/live_lab/` only after networking is confirmed.

## Connectivity Tests From Each VM Later

After VM creation in a later phase, test the server endpoints from each VM:

- controller health: `http://192.168.56.1:8020/health`
- final API readiness: `http://192.168.56.1:8014/ready`
- dashboard health: `http://192.168.56.1:8013/health`
- bridge readiness: `http://192.168.56.1:8016/ready`
- validator readiness: `http://192.168.56.1:8015/ready`
- Prometheus: `http://192.168.56.1:9090`
- Grafana: `http://192.168.56.1:3000`

## Next Step

P16.4 should clone the repository or copy `experiments/live_lab/` into the VMs,
then run dry-run node registration and connection checks. VM creation and OS
installation remain manual and outside this commit.

