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
