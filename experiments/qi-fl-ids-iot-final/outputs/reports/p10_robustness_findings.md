# P10 Robustness Findings

## Full label-flip results, poison_rate=0.2, poisoned_clients=1

- `fedavg`: macro_f1=0.9344680692677046, attack_recall=0.9185252525252525, fpr=0.04788888888888889, accuracy=0.9345185185185185, robustness_score=0.7332138326136503
- `fedavg_qga`: macro_f1=0.939086958704671, attack_recall=0.9368484848484848, fpr=0.058222222222222224, accuracy=0.9391957671957673, robustness_score=0.7389535803624364
- `qifa`: macro_f1=0.9375599645294771, attack_recall=0.9425454545454546, fpr=0.0676, accuracy=0.9377142857142857, robustness_score=0.738023618628375
- `qifa_qga`: macro_f1=0.9455302294225977, attack_recall=0.9565858585858585, fpr=0.06626666666666667, accuracy=0.9457037037037037, robustness_score=0.7464875389537231

Interpretation:

- QIFA+QGA is the best global result for Macro-F1, attack recall, accuracy, and robustness score.
- FedAvg has the best FPR, but it also has the weakest attack recall.
- QGA improves FedAvg under poisoning.
- QIFA improves attack detection but increases FPR.
- QIFA+QGA gives the best robustness/detection compromise.

The `fedavg` line with `macro_f1=0.4787` is a smoke readiness run and is not included in scientific conclusions.

Best full method: `qifa_qga`.

## Clean vs Poisoned

Clean comparison rows: 4.

## Warnings

- p10_qifa_weights_under_attack.png is a placeholder because the P10 in-process summaries do not expose QIFA per-round weights.
