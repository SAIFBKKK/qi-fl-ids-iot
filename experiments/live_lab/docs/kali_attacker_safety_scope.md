# Kali Attacker Safety Scope

`lab-attacker-kali` is planned as a lab workstation inspired by CICIoT2023. It
is not a repository automation target for real attacks.

## Policy

- Kali is used as a demonstration workstation for controlled lab scenarios.
- CICIoT2023 tool names are documented only as scientific references.
- The repository does not contain executable offensive command lines.
- The scripts in this repository do not launch real attacks.
- The lab uses controlled replay, feature windows, local controlled pcap files,
  or safe simulation.

## P16.3 Scenarios

- `icmp_flood_like`
- `tcp_syn_recon_like`
- `http_slow_like`

## Boundaries

P16.3 does not create the Kali VM, start packet capture, generate live traffic,
or run any scenario. The VM will remain powered off outside controlled
demonstration windows once it exists.

