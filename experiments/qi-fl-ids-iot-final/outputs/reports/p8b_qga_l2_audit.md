# P8-b QGA L2 Audit

## Files Inspected

- `experiments/qi-fl-ids-iot-final/src/fl_hierarchical/`
- `experiments/qi-fl-ids-iot-final/configs/hierarchical_flower.yaml`
- `experiments/qi-fl-ids-iot-final/outputs/hierarchical_flower/l2_family/`
- `experiments/qi-fl-ids-iot-final/src/qga/`
- `experiments/qi-fl-ids-iot-final/configs/qga_feature_selection.yaml`
- `experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/`
- `experiments/qi-fl-ids-iot-final/outputs/qga_fedavg_flower_l1/`
- `experiments/qi-fl-ids-iot-final/outputs/preprocessed/l2_family/`
- `experiments/qi-fl-ids-iot-final/outputs/partitions/l2_family/alpha_0.5/k3`

## P6 L2 Baseline

P6 provides a true Flower runtime for L2/L3 and uses P3 L2 `index_only` partitions. This is the safest reusable base for P8-b because it already protects the global test holdout and supports Windows manual/subprocess runtime.

## L2 Data

The L2 dataset is attack-only and contains scaled global `train`, `val`, and `test` NPZ files. Client partitions are stored as row-id indexes, so P8-b must avoid materializing client NPZ copies.

## Family Mapping

The L2 mapping contains 8 families: BruteForce, DDoS, DoS, Malware, Mirai, Recon, Spoofing, and Web-based.

## Reuse

Reusable from P8 L1: QGA profile sweep concepts, final mask structure, output contracts, and true Flower runtime discipline.

Reusable from P6: L2 loaders, model factory, metrics, one-vs-rest metrics, and manual Flower runtime patterns.

## Risks

- L2 arrays are large, so full QGA sweep and Flower validation can be expensive.
- L2 uses `index_only` partitions; client loading must map row IDs without duplicating data.
- Mask selection must not use global test.

## Recommendation

Build P8-b as a separate package `src/qga_l2`, reuse P6 loaders and metrics, and keep all final FL validation on true Flower.
