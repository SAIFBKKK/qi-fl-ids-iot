# Kaggle Dataset Plan - QI-FL-IDS-IoT

## Goal

Move dataset material out of GitHub and into a Kaggle dataset package that supports reproducible experiments without mixing data, models, logs, and generated reports.

## Current Dataset Findings

| Path | Size/role | Recommendation |
| --- | --- | --- |
| `data/balancing_v3_fixed300k_outputs/balancing_v3_fixed300k_balanced.parquet` | 929.97 MB balanced CICIoT2023 export | Primary Kaggle file. |
| `data/balancing_v3_fixed300k_outputs/balancing_v3_fixed300k_balanced.csv` | 2,285.77 MB duplicate CSV | Optional Kaggle file only if CSV users need it; prefer parquet to reduce size. |
| `data/balancing_v3_fixed300k_outputs/label_mapping.json` | small metadata | Include in Kaggle metadata. |
| `experiments/qi-fl-ids-iot-final/outputs/preprocessed/` | 2,208.36 MB L1/L2 preprocessed splits | Include only if the Kaggle dataset is meant to support exact reproduction without rerunning preprocessing. |
| `experiments/qi-fl-ids-iot-final/outputs/partitions/` | 1,101.82 MB FL Dirichlet partitions | Include only curated final split partitions, not all calibration grids. |
| `experiments/fl-iot-ids-v3/data/raw` and `data/processed` | 7.5 GB scenario data | Do not duplicate if final Kaggle package already contains final splits; otherwise archive as legacy dataset pack. |
| `data/cic-iot-2023/demo_subsets/*.parquet` | small tracked demo subsets | May remain in GitHub if licensing allows, or move to Kaggle under `samples/`. |
| `data/NSL-KDD/` | NSL-KDD notebooks/data/figures | Not part of final CICIoT2023 framework; remove from main repo or create separate legacy dataset reference. |
| `data/Federated-Learning-Based-Intrusion-Detection-System-main/` | UNSW-NB15 CSVs | Not final dataset; remove from main repo or external legacy archive. |
| `data/fl.pcap` | small packet capture | Treat as live-lab sample only after privacy review; otherwise externalize. |

## What Should Go to Kaggle

Recommended Kaggle dataset package:

- Balanced CICIoT2023 processed source in parquet.
- Label mapping and class metadata.
- Feature names for the 28-feature representation.
- Final QGA selected 12-feature list and mask.
- Preprocessing summary and leakage-prevention notes.
- Optional final L1 train/val/test splits if size is acceptable.
- Optional final FL client partitions for the retained public scenario only.
- Checksums and manifests for every data file.
- Dataset README with provenance, citation, schema, and reproduction commands.

## What Should Not Go to Kaggle

- Model checkpoints: `.pth`, `.pt`.
- Scalers as pickles unless strictly needed; prefer JSON scaler parameters or document regeneration.
- MLflow stores.
- Runtime logs.
- Generated report PDFs/PowerPoints.
- Docker deployment bundles.
- Private lab evidence.
- NSL-KDD/UNSW data unless this is a separate dataset package.
- Notebook drafts and old experiment outputs.

## Should Processed Train/Val/Test Files Be Uploaded?

Yes, but only for the final reproducibility package and only with a clear structure:

- Include final L1 binary train/val/test splits if the goal is exact reproduction of the final selected model.
- Include final L2/family splits only if public results depend on them.
- Include FL partitions only for the final retained scenario(s), not every alpha/K calibration output.
- Include manifests showing split ratios, row counts, feature count, label distribution, and whether test data was excluded from client training.
- Use parquet or compressed NumPy only if Kaggle size and usability are acceptable.

## Metadata Needed

Each Kaggle release should include:

- `dataset-metadata.json` for Kaggle.
- `README.md` dataset card.
- `CITATION.md` or citation section in README.
- `checksums.sha256`.
- `schema/feature_names_28.json`.
- `schema/selected_qga_features_12.json`.
- `schema/label_mapping.json`.
- `preprocessing/preprocessing_summary.json`.
- `splits/split_manifest.json`.
- `LICENSE` or license note reflecting the original CICIoT2023 terms.

## CICIoT2023 Citation

Use the citation already present in the repository README:

```text
Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A.,
Lu, R., and Ghorbani, A. A. (2023).
CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks
in IoT Environments. Sensors, 23(13), 5941.
DOI: 10.3390/s23135941
```

The Kaggle README should also acknowledge the Canadian Institute for Cybersecurity and link to the original dataset/source page if redistribution terms allow it.

## Proposed Kaggle Folder Structure

```text
qi-fl-ids-iot-ciciot2023/
  README.md
  dataset-metadata.json
  CITATION.md
  checksums.sha256
  LICENSE_OR_TERMS.md
  raw_processed/
    balancing_v3_fixed300k_balanced.parquet
    label_mapping.json
  schema/
    feature_names_28.json
    selected_qga_features_12.json
    qga_feature_mask_12.json
    label_to_binary.json
    label_to_family.json
  preprocessing/
    preprocessing_summary.json
    scaler_summary.json
    leakage_prevention.md
  splits/
    final_l1_binary/
      train.parquet
      val.parquet
      test.parquet
      split_manifest.json
    final_l2_family/
      train.parquet
      val.parquet
      test.parquet
      split_manifest.json
  federated_partitions/
    final_l1_alpha_0.5_k3/
      client_1_train.parquet
      client_1_val.parquet
      client_2_train.parquet
      client_2_val.parquet
      client_3_train.parquet
      client_3_val.parquet
      partition_manifest.json
  samples/
    demo_subsets/
      normal_traffic.parquet
      ddos_burst.parquet
      dos_slow.parquet
      mirai_wave.parquet
      recon_scan.parquet
      mixed_chaos.parquet
```

## Kaggle README Contents

The Kaggle README should describe:

- What this dataset is and why it exists.
- Relationship to original CICIoT2023.
- Exact preprocessing performed.
- Feature schema.
- Label schema.
- Final 12 QGA selected features.
- Train/val/test split policy.
- Federated partition policy.
- Leakage-prevention statement.
- Known limitations.
- How to use with this GitHub repo.
- Citation and acknowledgements.

## Upload Preparation Commands for Later

Dry-run staging inventory:

```powershell
Get-ChildItem data\balancing_v3_fixed300k_outputs -Force
Get-ChildItem experiments\qi-fl-ids-iot-final\outputs\preprocessed -Force -Recurse -File |
  Select-Object FullName, Length
```

Checksum generation:

```powershell
Get-ChildItem kaggle_dataset -Recurse -File |
  Get-FileHash -Algorithm SHA256 |
  ForEach-Object { "$($_.Hash)  $($_.Path)" } |
  Set-Content kaggle_dataset\checksums.sha256
```

Kaggle upload should be done later only after licensing review and explicit approval.
