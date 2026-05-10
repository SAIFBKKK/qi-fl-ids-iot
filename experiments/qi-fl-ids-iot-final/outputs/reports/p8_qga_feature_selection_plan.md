# P8 QGA Feature Selection Plan

## 1. Scope

P8 is limited to L1 binary IDS feature selection. It learns one QGA mask on L1 train/validation data, freezes the mask, then prepares/evaluates FedAvg L1 and HeteroFL L1 with the reduced feature set. L2 is deferred to P8-b.

## 2. P8.1 QGA standalone L1

- Load L1 train/val NPZ from P2.
- Load canonical 28 feature names.
- Exclude global test from optimization.
- Run quantum-inspired theta-vector mask search.
- Enforce `8 <= selected_features <= 24`.
- Evaluate masks with a fast MLP proxy.
- Save `selected_features.json`, `feature_mask.json`, `feature_ranking.csv`, `qga_history.csv`, `fitness_best.json`, and figures.

## 3. Fitness

Default objective:

`Fitness(m) = 0.6 * MacroF1(m) + 0.3 * Recall_attack(m) - 0.1 * (|F*| / 28)`

Justification:

- `alpha=0.6`: Macro-F1 is the main balanced IDS metric.
- `beta=0.3`: attack recall is critical because missed attacks are costly.
- `lambda=0.1`: feature penalty encourages lighter IoT models without dominating security metrics.
- IDS performance receives 90 percent of the weight, while feature reduction receives 10 percent.

## 4. P8.2 FedAvg L1 + QGA

- Load the latest QGA mask.
- Slice client train/val NPZ features for `alpha=0.5, K=3`.
- Train/evaluate the FedAvg L1 path with the reduced input dimension.
- Compare with P5 FedAvg baseline.
- Full 30-round run is documented but not launched automatically.

## 5. P8.3 HeteroFL L1 + QGA

- Load the latest QGA mask.
- Slice L1 partition arrays before constructing weak/medium/powerful models.
- Use the same HeteroFL prefix-slicing principle as P7 with reduced input dimension.
- Compare with P7 HeteroFL baseline.
- Full 30-round run is documented but not launched automatically.

## 6. P8-b L2 + QGA

L2 QGA is future work after L1 validation. It will reuse the L2 attack-only dataset and P3 L2 index-only partitions.

## 7. Validation commands

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/08_verify_qga_setup.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml
python experiments/qi-fl-ids-iot-final/src/scripts/08_run_qga_feature_selection.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml --mode smoke
python experiments/qi-fl-ids-iot-final/src/scripts/08_run_qga_fedavg_l1.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml --mode smoke --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000
python experiments/qi-fl-ids-iot-final/src/scripts/08_run_qga_heterofl_l1.py --config experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml --mode smoke --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000
```

## 8. Acceptance

P8 code-ready is accepted when verify passes, smoke QGA generates a valid mask and history, adapters are importable/runnable in smoke mode, tests pass, and the global test holdout is not used for selection.
