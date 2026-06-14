# Live Lab Runtime Artifacts

This directory contains only small runtime artifacts that can be cloned by the live lab VMs.

Current artifact:

- `l1_binary_robust_scaler.json`: lightweight JSON export derived from the final L1 binary RobustScaler.

Rules:

- Do not store datasets here.
- Do not store pcap files here.
- Do not store model checkpoints here.
- Do not store pickle files here.
- The source scaler pickle remains in the final experiment outputs and is not committed for VM runtime use.
