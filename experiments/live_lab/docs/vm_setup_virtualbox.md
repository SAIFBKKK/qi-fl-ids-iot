# VirtualBox VM Setup Plan

P16.1 step 0 does not create VMs. The VMs will be created later on drive `E:`
because drive `C:` does not have enough free space.

Planned VM root:

`E:\VirtualBox VMs\qi-fl-ids-iot-live-lab\`

Planned VMs:

- `iot-rpi-weak`: raspberry-like weak node.
- `iot-smart-watch-medium`: smart-watch-like medium node.

Recommended later setup:

- attach both VMs to the same isolated host-only or internal lab network;
- keep the server PC reachable on the configured lab IP;
- clone the repository or copy `experiments/live_lab/` into each VM;
- run only dry-run checks first.

