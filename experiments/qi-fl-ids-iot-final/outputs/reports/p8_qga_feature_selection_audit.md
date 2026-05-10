# P8 QGA Feature Selection Audit

## 1. Files audited

| Area | Path | Decision |
|---|---|---|
| v3 QGA selector | `experiments/fl-iot-ids-v3/src/qi/feature_selection.py` | Reuse ideas only |
| v3 theta QI selector | `experiments/fl-iot-ids-v3/src/qi/qi_feature_selector.py` | Reuse theta-vector search pattern |
| v3 QGA CLI | `experiments/fl-iot-ids-v3/src/scripts/run_qi_feature_selection.py` | Reimplement for final P8 paths |
| v3 QGA config | `experiments/fl-iot-ids-v3/configs/qi/qga_feature_selection.yaml` | Reuse smoke/full split idea |
| v3 QGA docs | `docs/qi/qga_feature_selection.md` | Reuse conceptual explanation |
| v3 QGA artifacts | `experiments/fl-iot-ids-v3/artifacts/qi_feature_selection/*/selected_features.json` | Historical only |
| P5 grid | `experiments/qi-fl-ids-iot-final/outputs/reports/p5_grid_summary.csv` | Baseline comparison source |
| P7 HeteroFL | `experiments/qi-fl-ids-iot-final/src/multitier_heterofl/` | Adapter source for P8 HeteroFL |
| Features | `experiments/qi-fl-ids-iot-final/outputs/artifacts/features/feature_names.json` | Canonical 28 feature list |
| L1 data | `experiments/qi-fl-ids-iot-final/outputs/preprocessed/l1_binary/*.npz` | QGA train/val source |
| L1 partitions | `experiments/qi-fl-ids-iot-final/outputs/partitions/l1_binary/alpha_0.5/k3` | FedAvg/HeteroFL evaluation source |

## 2. Existing QGA logic

The v3 code contains two useful patterns:

- `feature_selection.py` implements a deterministic filter/GA-style selector with mask repair, feature scores, redundancy penalty, and report generation.
- `qi_feature_selector.py` implements the more relevant quantum-inspired theta-vector method where each feature has a probability `p_i = sin(theta_i)^2`, masks are sampled from that vector, repaired, evaluated with a small MLP, and theta moves toward the best mask.

The historical v3 selector used an exact `k_features=15` constraint and multiclass-oriented settings. P8 final needs a bounded variable-size L1 binary mask, not a fixed-size v3 mask.

## 3. Reusable elements

- Theta-vector quantum-inspired mask sampling.
- Mask repair logic to avoid invalid masks.
- Smoke/full parameter split.
- Artifact naming style: `selected_features.json`, `feature_mask`, and a report.
- Applying selected feature indices at client startup rather than rewriting the source dataset.

## 4. Elements to reimplement

- Fitness must follow the P8 IDS formula: `0.6 * MacroF1 + 0.3 * Recall_attack - 0.1 * (features_count / 28)`.
- P8 must use only final L1 train/validation data to learn the mask.
- The global test holdout must be excluded from mask selection.
- Output contract must match final experiment conventions: rich `run_summary.json`, figures, logs, latest pointers, and criteria flags.
- FedAvg and HeteroFL adapters must use final P5/P7 outputs and final paths.

## 5. Historical QGA artifacts found

Two v3 artifacts were found:

- `normal_noniid`: 15 selected features, historical macro-F1 around `0.0401`.
- `absent_local`: 15 selected features, historical macro-F1 around `0.0265`.

These are not reusable as final masks because they were produced for v3 settings, exact-K feature selection, and historical scenarios. They are useful only as evidence that the theta-vector QGA path was implemented and exercised.

## 6. Risks

- A mask that is too aggressive can reduce Macro-F1 or increase FPR.
- A fixed `k_features` policy from v3 would be inconsistent with P8, which uses `min_features=8` and `max_features=24`.
- Fast MLP fitness is a proxy; full FedAvg/HeteroFL validation is still required for scientific conclusions.
- The stochastic optimizer needs deterministic seeding and full history export.
- Test leakage would invalidate the feature-selection result; P8 explicitly prevents test use during mask selection.

## 7. Recommendation

Implement a final P8 QGA package using the v3 theta-vector idea, but with a new L1 binary fitness function, bounded feature count, deterministic smoke/full modes, final output contract, and adapters for P5 FedAvg and P7 HeteroFL. Keep L2 QGA as a documented future P8-b extension.
