# VM Network Static IP Plan

Use a VirtualBox Host-Only Network for the local live lab.

## Static Addresses

- PC server: `192.168.56.1`
- VM1 `iot-drone-sitl`: `192.168.56.101`
- VM2 `iot-smart-watch-medium`: `192.168.56.102`
- VM3 `lab-attacker-kali`: `192.168.56.103`
- Subnet: `255.255.255.0`
- Gateway: none for Host-Only adapter

Adapter 2 NAT may be enabled temporarily for OS package installation only.

## Ports to Test From Each VM

- `1883`: MQTT Mosquitto
- `8013`: dashboard P13
- `8014`: final IDS API
- `8015`: online validator
- `8016`: final MQTT bridge
- `8020`: live lab controller
- `9090`: Prometheus
- `3000`: Grafana

No VM is created in P16.3.


