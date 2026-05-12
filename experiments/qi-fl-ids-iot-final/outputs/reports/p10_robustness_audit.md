# P10 Robustness Audit

## Files Audited

- `experiments/qi-fl-ids-iot-final/src/fl_l1/`
- `experiments/qi-fl-ids-iot-final/src/fl_l1_flower/`
- `experiments/qi-fl-ids-iot-final/src/qga/`
- `experiments/qi-fl-ids-iot-final/src/qifa/`
- `experiments/qi-fl-ids-iot-final/configs/fl_l1_fedavg.yaml`
- `experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml`
- `experiments/qi-fl-ids-iot-final/configs/qifa_l1.yaml`
- `experiments/qi-fl-ids-iot-final/outputs/reports/p5_grid_summary.csv`
- `experiments/qi-fl-ids-iot-final/outputs/reports/p8_qga_ablation_summary.csv`
- `experiments/qi-fl-ids-iot-final/outputs/reports/p9_qifa_ablation_summary.csv`
- `experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/final_selected_mask/`

## Findings

P5 provides a stable in-process FedAvg L1 runner and summary artifacts for alpha/K grid analysis. P8 provides the calibrated QGA L1 mask, and P9 provides the true Flower QIFA runtime and ablation outputs. These are enough to define defensive robustness scenarios without modifying existing datasets or previous phase outputs.

The safest P10 implementation is an additive package that applies poisoning only to in-memory copies of local client training partitions. Validation partitions remain clean by default, and the global test holdout remains server-side and unmodified.

## Risks

- Full Flower robustness runs can be expensive if every method, attack, and rate is launched automatically.
- Label poisoning can produce unstable one-round smoke metrics, so smoke results are readiness checks only.
- QIFA under attack should be interpreted with logged client weights and probabilities, not only final test metrics.

## Recommendation

Add P10 as a controlled defensive study with a lightweight smoke runner and report builder. Keep full FedAvg/QGA/QIFA/QIFA+QGA robustness scenarios manual and sequential.
