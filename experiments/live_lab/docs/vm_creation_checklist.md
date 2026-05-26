# VM Creation Checklist

P16.3 is a plan only. Do not create VMs automatically in this step.

## Storage

- Create folder later:
  `E:\VirtualBox VMs\qi-fl-ids-iot-live-lab\`

## VM Creation

- Create VM1 `iot-rpi-weak`.
- Create VM2 `iot-smart-watch-medium`.
- Create VM3 `lab-attacker-kali`.

## Network

- Adapter 1: VirtualBox Host-Only.
- Adapter 2: NAT, optional during installation only.

## OS Setup

- Install Ubuntu Server minimal on `iot-rpi-weak`.
- Install Ubuntu Server minimal on `iot-smart-watch-medium`.
- Install Kali Linux on `lab-attacker-kali`.
- Set hostname for each VM.
- Configure static IP for each VM.

## Connectivity

- Test server reachability from each VM.
- Test HTTP endpoints from each VM after server stack is intentionally running.
- Confirm MQTT broker port from each VM in a later step.

## Boundary

Do not launch packet capture or scenarios in this step.

