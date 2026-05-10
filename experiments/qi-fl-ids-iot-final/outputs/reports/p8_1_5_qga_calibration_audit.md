# P8.1.5 QGA Calibration Audit

## 1. Files audited

| Area | Path | Finding |
|---|---|---|
| QGA config | `experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml` | Current P8 has one balanced QGA profile and one final mask. |
| QGA package | `experiments/qi-fl-ids-iot-final/src/qga/` | Standalone QGA, Flower runtime, in-process helpers, reporting and tests exist. |
| QGA standalone outputs | `experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/` | Latest full mask is `run_20260508_104311`, 9 selected features. |
| QGA Flower outputs | `experiments/qi-fl-ids-iot-final/outputs/qga_fedavg_flower_l1/` | Full true-Flower run `run_20260508_111701` completed 30 rounds. |
| Ablation report | `experiments/qi-fl-ids-iot-final/outputs/reports/p8_qga_ablation_summary.csv` | Corrected to use full Flower and not smoke/in-process FedAvg. |
| Scripts | `experiments/qi-fl-ids-iot-final/src/scripts/08_*` | Existing QGA standalone and Flower FedAvg runtime are ready. |

## 2. Current QGA parameters

The current full QGA uses:

- population size: 20
- generations: 20
- mutation rate: 0.05
- min/max features: 8/24
- fitness: `0.6 * MacroF1 + 0.3 * Recall_attack - 0.1 * feature_ratio`
- seed: 42
- fast MLP validation on L1 train/validation only

## 3. Current result

The latest standalone full QGA mask selects 9 features:

`flow_duration`, `Protocol Type`, `Rate`, `urg_count`, `rst_count`, `HTTPS`, `ICMP`, `IAT`, `Number`.

The associated full true-Flower FedAvg run reaches Macro-F1 around `0.9448` on the final global holdout, with `true_flower_runtime=true`.

## 4. Limitations of a single mask

A single QGA run can overfit to one seed, one fitness weighting, and one validation distribution. Because QGA is stochastic and the final FL behavior depends on non-IID partitioning, one mask is not enough to freeze the final P8 engineering decision.

## 5. Need for multiple profiles

The calibration needs several profiles:

- balanced current profile for continuity;
- conservative profile to protect accuracy and Macro-F1;
- FPR-aware profile because false positives matter operationally in IDS;
- compression profile to test stronger feature reduction.

## 6. Need to integrate FPR

The current fitness optimizes Macro-F1 and attack recall but does not directly penalize false positives. The ablation shows FPR is a decisive operational metric, so calibration adds an optional `fpr_penalty`.

## 7. Need for multiple alpha/K scenarios

The previous FedAvg grid showed that alpha and client count strongly affect robustness. P8.1.5 therefore evaluates short true-Flower validation runs on:

- alpha=0.1, K=3
- alpha=0.5, K=3
- alpha=5.0, K=3

Then it extends the best mask to K=4 and K=5 for alpha=0.5.

## 8. Recommendation

Keep the current P8 mask as a strong candidate, but add a calibration sweep over profiles and seeds, rank masks using validation-only true-Flower short runs, and copy the final engineering choice into `outputs/qga_feature_selection/final_selected_mask/`.
