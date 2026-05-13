# P11 - FedTN/MPS Compression Study

## Objective

P11 studies tensor-network-inspired compression for the L1 IDS models after QGA feature selection. The goal is to reduce parameter count, model size, and estimated communication bandwidth while keeping performance close to the validated L1 baselines.

## Why Compress After QGA

QGA already reduces the input dimension from 28 to 12 using the calibrated `conservative_seed_42` mask. FedTN/MPS compression is then applied to hidden dense layers, which are the dominant parameter source after feature selection.

## Method

The implementation uses an MPS-style low-rank factorization for `fc1` and `fc2`. A dense layer is replaced by two smaller factors with configurable rank.

Ranks tested:

- 4
- 8
- 16
- 32

## Metrics

P11 reports:

- dense_num_parameters
- compressed_num_parameters
- dense_model_size_bytes
- compressed_model_size_bytes
- compression_ratio
- bandwidth_reduction_ratio
- Macro-F1 / attack recall / FPR if checkpoint evaluation is available

## Dry-run Result

The code-ready phase supports dry-run compression accounting without checkpoint loading. This is useful because checkpoints and run logs are intentionally not committed.

For `fedavg_qga` at rank 8:

| metric | dense | compressed |
|---|---:|---:|
| num_parameters | 10050 | 2978 |
| model_size_bytes | 40200 | 11912 |
| bandwidth_total_bytes | 7236000 | 2144160 |

The structural compression ratio is `0.2963`, corresponding to an estimated parameter and bandwidth reduction of about `70.37%`. Metric evaluation is not included in the dry-run because no checkpoint is loaded.

## Commands

Verify:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/11_verify_fedtn_mps_setup.py --config experiments/qi-fl-ids-iot-final/configs/fedtn_mps_l1.yaml
```

Dry-run:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/11_run_fedtn_mps_compression.py --config experiments/qi-fl-ids-iot-final/configs/fedtn_mps_l1.yaml --base-model fedavg_qga --rank 8 --dry-run
```

Build report:

```powershell
python experiments/qi-fl-ids-iot-final/src/scripts/11_build_fedtn_mps_report.py --config experiments/qi-fl-ids-iot-final/configs/fedtn_mps_l1.yaml
```

## Limits

The L1 QGA model is already compact, so gains are moderate in absolute terms. Higher ranks may be less compressive or even larger than the dense model. Checkpoint evaluation is required before presenting compressed models as final performance evidence.
