# P12 Final Findings

## Evidence Policy

P12 consolidates existing reports only. No full FL, Flower, robustness grid, Docker, dashboard, P13, or training run is launched.

Measured rows are final full evidence. P11 FedTN/MPS rows are dry-run structural estimates and must not be interpreted as measured Macro-F1, recall, or FPR.

## Final Objective Ranking

- Best production L1 compromise: P8 FedAvg + QGA.
- Best attack recall: P9 QIFA + QGA.
- Best FPR: P9 QIFA.
- Best poisoning robustness: P10 QIFA + QGA.
- Best structural compression: P11 FedTN/MPS rank 8.
- L2 experimental direction: P8-b QGA L2.

## Global Summary

Rows consolidated: 14.

## Figures

- C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\figures\p12_ablation\p12_l1_macro_f1_comparison.png
- C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\figures\p12_ablation\p12_l1_attack_recall_fpr_tradeoff.png
- C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\figures\p12_ablation\p12_l1_bandwidth_comparison.png
- C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\figures\p12_ablation\p12_qga_qifa_contribution_summary.png
- C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\figures\p12_ablation\p12_l2_p6_vs_p8b.png
- C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\figures\p12_ablation\p12_robustness_under_poisoning.png
- C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\figures\p12_ablation\p12_compression_size_reduction.png
- C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\figures\p12_ablation\p12_final_method_ranking_table.png
- C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\figures\p12_ablation\p12_best_model_by_objective.png

## Limitations

- P11 has no checkpoint-based measured Macro-F1/Recall/FPR in the current evidence pack.
- Multi-tier and L2 rows are experimental and not dashboard production candidates.
- Dashboard recommendation remains L1 production, with P8 FedAvg + QGA as deployment-oriented reference and P9/P10 as robustness evidence.
