# P16.1 Live Lab Realtime Structure

This directory is the VM-clonable live lab workspace for the final
Quantum-Inspired Federated Learning IoT IDS demo.

It is intentionally safe at step 0:

- no Docker startup;
- no VM creation;
- no packet capture;
- no active traffic generation;
- no training or Flower runtime;
- no changes to P8-P16 outputs.

## Target Lab

- Server PC: Docker Desktop stack from
  `experiments/qi-fl-ids-iot-final/deployment/docker-compose.final.yml`.
- VM1: `iot-rpi-weak`, raspberry-like weak node, server-side inference.
- VM2: `iot-smart-watch-medium`, smart-watch-like medium node, future edge inference.
- VM disk location planned for later:
  `E:\VirtualBox VMs\qi-fl-ids-iot-live-lab\`.

## Safe Data Path

Step 0 prepares code and docs for:

`packet window -> 28 features -> scaler -> selected 12 QGA features -> IDS`

The current runnable commands are dry-runs only. Live capture, VM creation, and
Docker startup are reserved for later steps.

