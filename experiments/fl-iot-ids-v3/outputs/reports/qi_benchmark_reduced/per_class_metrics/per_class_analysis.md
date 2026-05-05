# Per-class IDS analysis annex

## 1. Objective

This annex adds class-level evidence for the E1-E8 reduced QI benchmark, complementing aggregate Macro-F1 with one-vs-rest metrics and confusion-matrix structure.

## 2. Method

Metrics were extracted from the existing best checkpoints for E1-E8 only; no training was relaunched. Each model was evaluated on the scenario validation set, using `global_val.npz` when present and otherwise the concatenation of `node1`, `node2`, and `node3` validation NPZ files. QGA-15 experiments applied the saved boolean feature mask before inference.

## 3. Definitions

For class c, TP_c = CM[c,c], FP_c = sum(CM[:,c]) - TP_c, FN_c = sum(CM[c,:]) - TP_c, and TN_c = total - TP_c - FP_c - FN_c. Undefined ratios are reported as NaN when their denominator is zero.

## 4. Metrics used

The annex reports per-class accuracy, precision, recall, F1 score, specificity, false-positive rate, false-negative rate, and support, plus raw and row-normalized 34-class confusion matrices.

## 5. Weakest classes overall

- Class 0 `Backdoor_Malware`: mean F1=0.1702
- Class 31 `Uploading_Attack`: mean F1=0.1843
- Class 29 `Recon-PortScan`: mean F1=0.2621
- Class 33 `XSS`: mean F1=0.3078
- Class 16 `DNS_Spoofing`: mean F1=0.3215

## 6. Rare classes analysis

- `normal_noniid`: none below support < 1000
- `absent_local`: none below support < 1000

## 7. QIFA per-class effect

- E6 - E5 (absent_local, 28f), class 13 `DDoS-TCP_Flood`: QIFA degrades F1 by -0.7650 and recall by -0.7046.
- E6 - E5 (absent_local, 28f), class 21 `DoS-UDP_Flood`: QIFA degrades F1 by -0.5885 and recall by -0.4615.
- E8 - E7 (absent_local, QGA-15), class 20 `DoS-TCP_Flood`: QIFA improves F1 by 0.2136 and recall by 0.1351.
- E8 - E7 (absent_local, QGA-15), class 8 `DDoS-PSHACK_Flood`: QIFA improves F1 by 0.1521 and recall by 0.1297.
- E6 - E5 (absent_local, 28f), class 12 `DDoS-SynonymousIP_Flood`: QIFA degrades F1 by -0.1510 and recall by -0.2692.

## 8. QGA-15 per-class effect

- E8 - E6 (absent_local, QIFA: QGA-15 - 28f), class 15 `DDoS-UDP_Fragmentation`: QGA-15 degrades F1 by -0.9870 and recall by -0.9804.
- E7 - E5 (absent_local, FedAvg: QGA-15 - 28f), class 15 `DDoS-UDP_Fragmentation`: QGA-15 degrades F1 by -0.9862 and recall by -0.9795.
- E7 - E5 (absent_local, FedAvg: QGA-15 - 28f), class 8 `DDoS-PSHACK_Flood`: QGA-15 degrades F1 by -0.8882 and recall by -0.9684.
- E7 - E5 (absent_local, FedAvg: QGA-15 - 28f), class 20 `DoS-TCP_Flood`: QGA-15 degrades F1 by -0.8408 and recall by -0.9398.
- E8 - E6 (absent_local, QIFA: QGA-15 - 28f), class 25 `Mirai-udpplain`: QGA-15 degrades F1 by -0.8198 and recall by -0.7211.

## 9. Most degraded classes

- E8 - E6 class 15 `DDoS-UDP_Fragmentation`: delta F1=-0.9870
- E7 - E5 class 15 `DDoS-UDP_Fragmentation`: delta F1=-0.9862
- E7 - E5 class 8 `DDoS-PSHACK_Flood`: delta F1=-0.8882
- E7 - E5 class 20 `DoS-TCP_Flood`: delta F1=-0.8408
- E8 - E6 class 25 `Mirai-udpplain`: delta F1=-0.8198

## 10. Most improved classes

- E8 - E6 class 13 `DDoS-TCP_Flood`: delta F1=0.4523
- E8 - E7 class 20 `DoS-TCP_Flood`: delta F1=0.2136
- E8 - E7 class 8 `DDoS-PSHACK_Flood`: delta F1=0.1521
- E6 - E5 class 25 `Mirai-udpplain`: delta F1=0.1105
- E6 - E5 class 3 `CommandInjection`: delta F1=0.1000

## 11. Limitations

These are single-seed, validation-set-only metrics. They reuse the best checkpoint selected during the original benchmark, so the analysis is suitable as a reference annex but not as a new independent benchmark. QGA and QIFA effects can also interact with class support, so rare-class conclusions should be treated as diagnostic rather than definitive.
