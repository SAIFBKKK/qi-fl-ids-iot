# P10 - Robustness and Poisoning Attacks

## Objective

P10 adds a defensive robustness study for L1 binary IoT IDS. It measures how controlled perturbations affect FedAvg, FedAvg + QGA, QIFA, and QIFA + QGA.

## Simulated Attacks

- `clean`: no perturbation.
- `label_flip`: flips a proportion of local train labels on compromised clients.
- `attack_to_normal`: changes a proportion of attack labels to normal on compromised clients.
- `feature_noise`: applies bounded Gaussian noise to local train features.

All attacks are applied only to in-memory copies of client training data. P3 partitions are never modified.

## Metrics

The study tracks macro-F1, attack recall, FPR, FNR, accuracy, confusion matrix, per-client metrics, round metrics, and a robustness score.

## Smoke Result

Smoke runs are readiness checks only. They use one FL round and a capped sample count, so they are not scientific evidence.

## Full label-flip results, poison_rate=0.2, poisoned_clients=1

The full scientific P10 scenario uses alpha=0.5, K=3, rounds=30, `label_flip`, `poison_rate=0.2`, and one poisoned client.

| method | macro_f1 | attack_recall | fpr | accuracy | robustness_score |
|---|---:|---:|---:|---:|---:|
| fedavg | 0.9344680692677046 | 0.9185252525252525 | 0.04788888888888889 | 0.9345185185185185 | 0.7332138326136503 |
| fedavg_qga | 0.939086958704671 | 0.9368484848484848 | 0.058222222222222224 | 0.9391957671957673 | 0.7389535803624364 |
| qifa | 0.9375599645294771 | 0.9425454545454546 | 0.0676 | 0.9377142857142857 | 0.738023618628375 |
| qifa_qga | 0.9455302294225977 | 0.9565858585858585 | 0.06626666666666667 | 0.9457037037037037 | 0.7464875389537231 |

QIFA+QGA is the best global result for Macro-F1, attack recall, accuracy, and robustness score. FedAvg has the best FPR, but it has the weakest attack recall. QGA improves FedAvg under poisoning. QIFA improves attack detection but increases FPR. QIFA+QGA gives the best robustness/detection compromise.

The older `fedavg` run with `macro_f1=0.4787` is a smoke readiness run and is not included in the scientific conclusions.

## Manual Commands

Verify:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/10_verify_robustness_setup.py --config experiments/qi-fl-ids-iot-final/configs/robustness_l1.yaml
```

Smoke:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/10_run_robustness_smoke.py --config experiments/qi-fl-ids-iot-final/configs/robustness_l1.yaml --method fedavg --attack-type label_flip --poison-rate 0.2 --poisoned-clients 1 --rounds 1 --max-samples 1000
```

Single manual scenario:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/10_run_robustness_scenario.py --config experiments/qi-fl-ids-iot-final/configs/robustness_l1.yaml --method qifa_qga --alpha 0.5 --clients 3 --rounds 30 --attack-type label_flip --poison-rate 0.2 --poisoned-clients 1
```

Build report:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/10_build_robustness_report.py --config experiments/qi-fl-ids-iot-final/configs/robustness_l1.yaml
```

## Limits

The in-process smoke runner validates poisoning and reporting. Final QIFA/QIFA+QGA evidence should be run through the true Flower runtime path before being used as final scientific evidence.
