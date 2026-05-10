# P8-b QGA L2 Plan

## P8-b.1 QGA L2 Standalone

Run profile/seed sweeps on L2 train/validation only. Produce candidate masks and validation metrics without Flower.

## P8-b.2 Short Flower Calibration

Evaluate the top masks with true Flower for 5 rounds on alpha values 0.1, 0.5, and 5.0 with K=3. Use validation only for ranking.

## P8-b.3 Final Mask Selection

Filter only true Flower 5-round validations with `test_sent_to_clients=false`, require at least 3 scenarios, and select by engineering score.

## P8-b.4 Full L2 FedAvg + QGA Flower

Run alpha=0.5, K=3, rounds=30 with a true Flower server and clients. The global test holdout is loaded only server-side for final evaluation.

## P8-b.5 L2 Ablation

Compare P6 L2 Flower baseline against P8-b L2 FedAvg + QGA Flower.

## Commands

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/08_b_verify_qga_l2_setup.py --config experiments/qi-fl-ids-iot-final/configs/qga_l2_feature_selection.yaml
python experiments/qi-fl-ids-iot-final/src/scripts/08_b_run_qga_l2_profile_sweep.py --config experiments/qi-fl-ids-iot-final/configs/qga_l2_feature_selection.yaml
python experiments/qi-fl-ids-iot-final/src/scripts/08_b_run_qga_l2_flower_smoke.py --config experiments/qi-fl-ids-iot-final/configs/qga_l2_feature_selection.yaml --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000 --address 127.0.0.1:8084
```
