# Centralized Baseline — CIC-IoT-2023

> **Experiment:** `baseline-CIC_IOT_2023`  
> **Purpose:** Establish centralized ML performance ceilings before Federated Learning  
> **Dataset:** CIC-IoT-2023 — 34 attack classes · 33 features · ~7.9M total samples  
> **Status:** ✅ Complete

---

## Overview

Before introducing Federated Learning and Quantum-Inspired optimization, this experiment establishes a **centralized reference point** — classical ML models trained on the full CIC-IoT-2023 dataset in a single-node setting. These baselines serve two roles in the PFE:

- **Performance ceiling** — the best a model can achieve with all data centralized
- **Comparison target** — FL + QI results in `fl-iot-ids-v1` and `fl-iot-ids-v2` are evaluated relative to these numbers

---

## Notebooks

The experiment is structured as three sequential notebooks. Run them in order.

### `CICIOT23_Pipeline.ipynb` — EDA & Preprocessing

Full exploratory analysis and data cleaning pipeline on the raw CIC-IoT-2023 CSVs. Produces all preprocessed artifacts consumed by downstream notebooks and by `fl-iot-ids-v1`.

**What it does:**

1. **Load** — Concatenates `train/`, `validation/`, `test/` CSV folders into three DataFrames (5.49M / 1.18M / 1.18M rows, 47 columns each)
2. **EDA** — Column types, NaN/Inf counts, duplicate detection, class distribution analysis, correlation heatmap, constant/near-constant feature identification
3. **Feature cleaning** — Drops 13 non-informative features:
   - Constant (1 unique value): `Telnet`, `IRC`
   - Near-constant (>99.9% same value): `Drate`, `ece_flag_number`, `cwr_flag_number`, `DNS`, `SMTP`, `SSH`, `DHCP`, `ARP`
   - Perfect correlates (r=1.0, keep more interpretable): `Srate` (→ keep `Rate`), `LLC` (→ keep `IPv`), `Radius` (→ keep `Std`)
4. **Inf handling** — Replace ±Inf with column median (fitted on train only)
5. **Deduplication** — Remove 134,565 duplicate rows from train only (val/test untouched)
6. **Label encoding** — Stable sorted 34-class integer mapping from train labels
7. **Class weights** — `balanced` strategy via `compute_class_weight`
8. **Scaling** — `RobustScaler` fitted on train only, applied to all splits (float32 cast)
9. **Save artifacts** — `.npz` arrays + 4 `.pkl` artifact files

**Key findings:**

| Finding | Detail |
|---|---|
| Raw features | 46 float + 1 label = 47 columns |
| Features after cleaning | 33 |
| NaN / Inf | None in raw data |
| Train duplicates removed | 134,565 |
| Class imbalance | BenignTraffic = 2.36% of train |
| Feature scale range | IAT ~8×10⁷, Rate ~8×10⁶ → RobustScaler mandatory |
| All splits consistent | No unseen classes in val or test |

---

### `Centralized_Baseline.ipynb` — Model Training & Evaluation

Hyperparameter search and final evaluation of 5 ML models on the preprocessed augmented dataset (5.72M training samples after benign augmentation).

**Pipeline:**

```
Load ciciot23_34class_augmented.npz
        ↓
Sanity checks (NaN, Inf, shape, clip ±1e6)
        ↓
Stratified subset — 400K rows for tuning
        ↓
RandomizedSearchCV + StratifiedKFold (3-fold)
        ├── LogisticRegression  (saga solver, multinomial)
        ├── SGDClassifier       (log_loss, elasticnet)
        ├── RandomForest        (300 trees, balanced_subsample)
        ├── XGBoost-GPU         (hist, CUDA, RTX 3050)
        └── MLPClassifier       (early stopping)
        ↓
Compare on validation set (Macro F1 primary metric)
        ↓
Re-train winner on full 5.72M train set
        ↓
Final evaluation on test set + per-class report
        ↓
Save model (.pkl) + metrics (.json)
```

**Design decisions:**

| Decision | Reason |
|---|---|
| Tune on 400K subset | 5.7M × CV × 5 models = days of compute |
| 3-fold CV | Safer memory/time budget on large data |
| Macro F1 as primary metric | Dataset imbalanced — accuracy is misleading |
| `class_weight="balanced"` | BenignTraffic underrepresented even after augmentation |
| Re-train on full data | Tuning subset finds best params; full data gives best model |

---

### `03b_Baseline_Fixes.ipynb` — Corrections

Fixes 4 errors found in the initial baseline results. Always use the `_fixed` JSON files for the report.

| # | Error | Fix |
|---|---|---|
| 1 | `attack_recall`/`benign_recall` were F1 scores, not recall | Recomputed with `recall_score()` |
| 2 | 7-family mapping contained fake labels (`BruteForce-Web`, `BruteForce-XSS`) | Rebuilt from real `id_to_label` |
| 3 | Storytelling claimed Binary macro F1 > 0.95 | Replaced with honest values |
| 4 | Summary table used wrong metric values | Rebuilt from corrected metrics |

---

## Results

### Model Comparison — Validation Set

| Rank | Model | Macro F1 | Accuracy |
|---|---|---|---|
| 🥇 | RandomForest | 0.5559 | 0.9593 |
| 🥈 | XGBoost-GPU | 0.5125 | 0.9272 |
| 🥉 | MLPClassifier | 0.3444 | 0.8561 |
| 4 | SGDClassifier | 0.1332 | 0.3216 |
| 5 | LogisticRegression | 0.0381 | 0.0528 |

> **Best model: RandomForest** — `max_depth=40`, `max_features=0.5`, `min_samples_split=2`

### Final Results — Test Set (Corrected)

| Setting | Accuracy | Macro F1 | Weighted F1 | Notes |
|---|---|---|---|---|
| 34-class (full) | 0.9518 | 0.5106 | 0.9547 | Primary metric |
| Binary (Benign/Attack) | 0.9820 | 0.6947 | — | Attack recall=0.9996, Benign recall=0.2527 ⚠️ |
| 7-Family (fixed) | 0.9535 | 0.5702 | 0.9551 | 8 families |

### Top-K Accuracy (Test Set)

| Top-K | Accuracy |
|---|---|
| Top-1 | 0.9518 |
| Top-3 | 0.9820 |
| Top-5 | 0.9928 |

### Key Weaknesses Identified

These are the **improvement targets** for `fl-iot-ids-v1` (FL) and `fl-iot-ids-v2` (QI):

| Class | Recall | F1 | Support | Issue |
|---|---|---|---|---|
| BenignTraffic | 0.501 | 0.592 | 27,709 | **Primary target** — 74.7% false positive rate |
| Recon-PingSweep | 0.000 | 0.000 | 53 | Too rare |
| CommandInjection | 0.000 | 0.000 | 119 | Too rare |
| VulnerabilityScan | 0.003 | 0.007 | 913 | Confused with other Recon |
| BruteForce family | 0.01–0.02 | 0.02–0.04 | small | Low support across all variants |
| WebAttack family | 0.01–0.05 | 0.02–0.10 | small | SqlInjection, XSS, BrowserHijacking |

**Most common confusions:**

| True | Predicted | Count |
|---|---|---|
| BenignTraffic | Uploading_Attack | 13,753 |
| Mirai-greip_flood | Mirai-greeth_flood | 11,029 |
| MITM-ArpSpoofing | Uploading_Attack | 3,917 |

---

## Artifacts Produced

All artifacts are saved to `results_baseline/` and `artifacts/`. These files are **required** by `fl-iot-ids-v1`.

## Extracted Utilities

During repository cleanup, the useful dataset-building utilities from
`baseline-CIC_IOT_2023_v2/src` were preserved inside the kept baseline:

| Script | Purpose |
|---|---|
| `src/training/build_level2_family_dataset.py` | Build the Level-2 family dataset from `post_balancing_preprocessing_FINAL_base_balanced_only/exports/*.csv` |
| `src/training/build_level3_subtype_datasets.py` | Build per-family Level-3 subtype datasets from the same preprocessing exports |

These scripts were kept because they add reusable hierarchy-building logic.
The old v2 flat trainer was not reintroduced because the current `balanced_v3`
training scripts remain the stronger baseline path.

### Preprocessed data (not versioned — generate locally)

| File | Location | Description |
|---|---|---|
| `ciciot23_34class_preprocessed.npz` | `processed/` | X/y arrays — 5.36M train, 1.18M val/test, 33 features |
| `ciciot23_34class_augmented.npz` | `processed/` | Same + benign augmentation → 5.72M train samples |

### Model artifacts (not versioned)

| File | Location | Description |
|---|---|---|
| `scaler_robust.pkl` | `artifacts/` | RobustScaler fitted on train — **required by fl-iot-ids-v1** |
| `label_mapping_34.pkl` | `artifacts/` | `label_to_id` / `id_to_label` dicts — **required by fl-iot-ids-v1** |
| `class_weights_34.pkl` | `artifacts/` | Balanced class weights for 34 classes — **required by fl-iot-ids-v1** |
| `feature_names.pkl` | `artifacts/` | List of 33 feature names after cleaning — **required by fl-iot-ids-v1** |
| `best_model_rf.pkl` | `results_baseline/` | Trained RandomForest (full train, best params) |

### Metrics (versioned in Git — see `results_baseline/`)

| File | Description |
|---|---|
| `metrics_test.json` | Global test metrics + Top-K accuracy |
| `metrics_binary_fixed.json` | Binary classification metrics (corrected) |
| `metrics_7family_fixed.json` | 7-family classification metrics (corrected mapping) |
| `classification_report.json` | Full per-class precision/recall/F1 |
| `all_models_val_results.json` | All 5 models validation scores |
| `per_class_metrics.csv` | Per-class metrics sorted by recall |
| `top_confusions.csv` | Top misclassification pairs |
| `paper_summary_fixed.csv` | Corrected paper-level summary table |

> ⚠️ Use `metrics_binary_fixed.json` and `metrics_7family_fixed.json` — the unfixed versions contain errors.

---

## How to Reproduce

### 1. Download and place dataset

```
experiments/baseline-CIC_IOT_2023/raw/
├── train/train.csv
├── validation/validation.csv
└── test/test.csv
```

See [`data/README.md`](../../data/README.md) for download instructions.

### 2. Create output directories

```bash
mkdir -p experiments/baseline-CIC_IOT_2023/processed
mkdir -p experiments/baseline-CIC_IOT_2023/artifacts
mkdir -p experiments/baseline-CIC_IOT_2023/results_baseline
```

### 3. Run notebooks in order

```
1. CICIOT23_Pipeline.ipynb          → produces processed/*.npz + artifacts/*.pkl
2. Centralized_Baseline.ipynb       → produces results_baseline/*.json + best_model_rf.pkl
3. 03b_Baseline_Fixes.ipynb         → produces *_fixed.json files (use these for the report)
```

> The notebooks require the path `ROOT` to be updated to your local machine. All paths are defined in Section 0 of each notebook.

### 4. Copy artifacts to fl-iot-ids-v1

After running the preprocessing notebook, copy the 4 required artifacts:

```bash
# From experiments/baseline-CIC_IOT_2023/artifacts/
cp scaler_robust.pkl      ../fl-iot-ids-v1/artifacts/
cp label_mapping_34.pkl   ../fl-iot-ids-v1/artifacts/
cp class_weights_34.pkl   ../fl-iot-ids-v1/artifacts/
cp feature_names.pkl      ../fl-iot-ids-v1/artifacts/
```

These files are already referenced in `fl-iot-ids-v1/src/common/paths.py` and loaded by the `Preprocessor` and `DataLoader` modules.

---

## Hardware Used

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop |
| CUDA | 12.1 |
| Python | 3.11 |
| XGBoost | GPU training via `device="cuda"`, `tree_method="hist"` |
| RandomForest/MLP/LR | CPU (`n_jobs=-1`) |

Approximate runtimes on this hardware:

| Step | Time |
|---|---|
| Preprocessing pipeline | ~5 min |
| LogisticRegression tuning (3 iter) | ~16.5 min |
| RandomForest tuning (8 iter, 400K) | ~16 min |
| XGBoost-GPU tuning (8 iter, 400K) | ~32.5 min |
| MLP tuning (8 iter, 400K) | ~12 min |
| RF re-train on full 5.7M | ~21 min |
| Binary classifier training | ~14 min |
| 7-family classifier training | ~22 min |

---

## Connection to fl-iot-ids-v1

```
baseline-CIC_IOT_2023/
│
├── artifacts/scaler_robust.pkl       ──────→  fl-iot-ids-v1/artifacts/scaler_robust.pkl
├── artifacts/label_mapping_34.pkl    ──────→  fl-iot-ids-v1/artifacts/label_mapping_34.pkl
├── artifacts/class_weights_34.pkl    ──────→  fl-iot-ids-v1/artifacts/class_weights_34.pkl
├── artifacts/feature_names.pkl       ──────→  fl-iot-ids-v1/artifacts/feature_names.pkl
│
└── results_baseline/
    ├── metrics_test.json             ──────→  FL baseline comparison reference
    ├── metrics_binary_fixed.json     ──────→  FL binary comparison reference
    └── paper_summary_fixed.csv       ──────→  PFE report Table 3
```

The FL experiment targets improvement on the 34-class Macro F1 (0.5106) — specifically BenignTraffic recall and rare class detection — while preserving the high accuracy on dominant DDoS/DoS/Mirai classes.

---

## References

- Koroniotis, N. et al. (2023). *A new network intrusion detection dataset for IoT: CIC-IoT-2023.* Sensors, 23(13), 5941. https://doi.org/10.3390/s23135941
- Bergstra, J. & Bengio, Y. (2012). *Random search for hyper-parameter optimization.* JMLR, 13, 281–305.
