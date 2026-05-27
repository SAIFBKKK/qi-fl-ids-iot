# Kali VM Readiness

## Role

`lab-attacker-kali` is the future controlled scenario workstation for the live lab. It is not used in P16.10 to run scenarios. It is only inventoried and checked for host-only network readiness.

## Expected Identity

- VM name: `lab-attacker-kali`
- Expected host-only IP: `192.168.56.103`
- Server IP: `192.168.56.1`
- Weak IoT node: `192.168.56.101`
- Medium IoT node: `192.168.56.102`

## Hostname And IP Checklist

- Confirm the VM hostname identifies the Kali workstation.
- Confirm the host-only interface has `192.168.56.103`.
- Confirm the VM is connected to the VirtualBox host-only network.
- Keep the expected IP stable for repeatable evidence collection.

## Server Access Checklist

Verify only the known fixed server endpoints:

- live-lab-controller health: `http://192.168.56.1:8020/health`
- final IDS API readiness: `http://192.168.56.1:8014/ready`
- final MQTT bridge readiness: `http://192.168.56.1:8016/ready`
- online-validator readiness: `http://192.168.56.1:8015/ready`
- dashboard health: `http://192.168.56.1:8013/health`

## Tool Inventory Checklist

P16.10 may inventory whether the following tools are installed. The readiness script does not execute them for traffic generation:

- hping3
- nmap
- python3
- curl
- tcpdump or tshark

## Limits

- No scenario execution in P16.10.
- No live capture in P16.10.
- No network scan in P16.10.
- No generated attack traffic in P16.10.
- No change to P8-P16.9.2 results.
