| method | features_count | macro_f1 | attack_recall | fpr | bandwidth_total_bytes | true_flower_runtime | calibration_decision_used | selected_mask_source | accepted |
|---|---|---|---|---|---|---|---|---|---|
| P5 FedAvg baseline | 28 | 0.9407415103639889 | 0.9474141414141414 | 0.06626666666666667 | 8710560 | False |  |  | True |
| P8 FedAvg + QGA Flower | 12 | 0.9479843808395867 | 0.955010101010101 | 0.059444444444444446 | 7236000 | True | True | final_selected_mask | True |
| P8 HeteroFL + QGA | 9 | 0.9118787405609233 | 0.9834949494949495 | 0.16435555555555556 | 11076000 | False |  |  | True |

## Warnings

- Ignored P8 FedAvg+QGA Flower smoke run(s): run_20260508_102929, run_20260508_150503, run_20260508_150537, run_20260508_150610
- Ignored in-process P8 FedAvg+QGA helper summary: C:/Users/saifb/dev/qi-fl-ids-iot/experiments/qi-fl-ids-iot-final/outputs/qga_fedavg_l1/alpha_0.5/k3/latest_run_summary.json
- Ignored non-calibrated P8 FedAvg+QGA Flower full run(s): run_20260508_111701, run_20260508_152046
