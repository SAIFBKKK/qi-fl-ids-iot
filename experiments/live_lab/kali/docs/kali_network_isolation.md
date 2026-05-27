# Kali Network Isolation

## Network Mode

The Kali workstation should use the VirtualBox host-only network for live lab observations.

## Lab IPs

- Server Windows host: `192.168.56.1`
- VM1 `iot-rpi-weak`: `192.168.56.101`
- VM2 `iot-smart-watch-medium`: `192.168.56.102`
- VM3 `lab-attacker-kali`: `192.168.56.103`
- Subnet: `255.255.255.0`

## Isolation Recommendation

For future controlled scenario windows, disable NAT on the Kali VM unless package installation is explicitly required outside the demonstration. During the live demonstration, avoid any external network path.

## Readiness Verification

- Verify the Kali VM has the expected host-only IP.
- Verify the server endpoints are reachable on the fixed ports.
- Verify VM1 and VM2 are reachable as fixed lab hosts.
- Verify the dashboard `/demo` page is reachable from the server browser or lab network.

## Safety Boundary

Do not scan the network. Do not broaden checks to a subnet. P16.10 only uses fixed lab IPs and fixed server endpoints for readiness evidence.
