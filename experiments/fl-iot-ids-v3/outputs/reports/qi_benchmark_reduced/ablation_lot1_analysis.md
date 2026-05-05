# Lot 1 Ablation: QIFA vs FedAvg (28 features)

## Scope
This analysis uses existing 30-round outputs only. No training was relaunched.
Experiments covered: E1, E2, E5, E6.

## Main Results
- Best final Macro-F1: `E1` (normal_noniid / FedAvg) = `0.7697173983054993`.
- Best rare-class recall: `E1` (normal_noniid / FedAvg) = `0.4441882616619788`.
- Lowest FPR: `E6` (absent_local / QIFA) = `0.061495524702539675`.

## QIFA vs FedAvg Deltas
- normal_noniid (E2 - E1): delta Macro-F1 = `-0.002467134906848578`, delta accuracy = `-0.00222095684096546`, delta FPR = `0.01910214557762248`, delta rare recall = `-0.0020484995385492977`, delta rare Macro-F1 = `-0.006810054464186921`.
- absent_local (E6 - E5): delta Macro-F1 = `-0.01777445656660992`, delta accuracy = `-0.012020414267692958`, delta FPR = `-0.0029526606236371794`, delta rare recall = `0.01671818515048018`, delta rare Macro-F1 = `0.01966178522153242`.

## Interpretation
- On `normal_noniid`, QIFA is `unfavorable` for Macro-F1 and `unfavorable` for FPR versus FedAvg.
- On `absent_local`, QIFA is `unfavorable` for Macro-F1, `favorable` for FPR, and `favorable` for rare recall versus FedAvg.

## QIFA Internal Metrics
- normal_noniid QIFA final diversity mean: `0.22825732615424885`, average diversity mean: `0.2602479092770499`.
- absent_local QIFA final diversity mean: `0.29334982777104557`, average diversity mean: `0.33273385083541296`.
- Perturbation was disabled in these runs if `qifa_perturbation_applied_sum` is `0.0`.

## Provisional Conclusion
For lot 1, conclusions are limited to 28-feature FedAvg vs QIFA. The QGA-15 communication/performance question requires lot 2 (E3/E4/E7/E8).
If QIFA improves rare recall or FPR while reducing Macro-F1, report that trade-off explicitly rather than treating QIFA as a universal win.

## Generated Files
- `ablation_lot1_qifa_vs_fedavg.csv`
- `ablation_lot1_qifa_deltas.csv`
- `ablation_lot1_qifa_internal.csv`
