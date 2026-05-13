# P11 FedTN/MPS Audit

## Files Audited

- `experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml`
- `experiments/qi-fl-ids-iot-final/configs/qifa_l1.yaml`
- `experiments/qi-fl-ids-iot-final/outputs/reports/p8_qga_ablation_summary.csv`
- `experiments/qi-fl-ids-iot-final/outputs/reports/p9_qifa_ablation_summary.csv`
- `experiments/qi-fl-ids-iot-final/outputs/reports/p10_robustness_full_summary.csv`
- `experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/final_selected_mask/`
- `experiments/qi-fl-ids-iot-final/src/qga/`
- `experiments/qi-fl-ids-iot-final/src/qifa/`
- `experiments/qi-fl-ids-iot-final/src/fl_l1/`
- `experiments/qi-fl-ids-iot-final/src/fl_l1_flower/`

## Available L1 Models

The final L1 lineage includes P5 FedAvg, P8 FedAvg+QGA, P9 QIFA, P9 QIFA+QGA, and P10 robustness outputs. The calibrated QGA mask is available as `conservative_seed_42` with 12 selected features.

## Compression Targets

P8 FedAvg+QGA is the production-oriented base for compression because it already reduces input features while preserving strong L1 performance. P9 QIFA+QGA is the robust base to compare because P10 showed it is the best method under label flipping.

## Limitations

The L1 QGA model is already small: 12 -> 128 -> 64 -> 2 is roughly 10k parameters. Tensor-network-inspired compression can reduce communication and model size, but the expected absolute gain is moderate and rank-sensitive. Checkpoint-based metric evaluation requires local checkpoints and is not assumed to be versioned.

## Recommendation

Implement P11 first as post-training structural compression/dry-run with ranks 4, 8, 16, and 32. Use rank 8 as the initial smoke. Add full checkpoint evaluation only when local P8/P9 checkpoints are explicitly available.
