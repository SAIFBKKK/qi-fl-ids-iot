# Data Directory

This directory contains **no raw dataset files** — all datasets are excluded from version control due to licensing restrictions and file size constraints. Only `.gitkeep` placeholder files and this documentation are tracked.

---

## Dataset Selection Process

Three candidate datasets were evaluated for this project. The selection was based on recency, attack diversity, IoT relevance, and suitability for federated learning simulation.

### Candidates Evaluated

#### 1. NSL-KDD

| Property | Value |
|----------|-------|
| Origin | Canadian Institute for Cybersecurity, 2009 |
| Size | ~125k train / ~22k test records |
| Features | 41 |
| Attack classes | 4 (DoS, Probe, R2L, U2R) |
| Format | `.arff` / `.csv` |

NSL-KDD is a refined version of the KDD Cup 1999 dataset, correcting duplicate record issues. It remains a standard benchmark for IDS research. However, its age (2009) and limited attack diversity make it unsuitable as the primary dataset for a modern IoT IDS.

**Role in this project:** Used for preliminary experimentation with the Quantum-Inspired Evolutionary Algorithm (QIEA) for feature selection — see [QIEA experiment](#qiea-feature-selection-nsl-kdd) below.

---

#### 2. UNSW-NB15

| Property | Value |
|----------|-------|
| Origin | UNSW Canberra Cyber Range Lab, 2015 |
| Size | ~257k records |
| Features | 49 |
| Attack classes | 9 (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms) |
| Format | `.csv` |

UNSW-NB15 was generated using IXIA PerfectStorm in a more realistic hybrid environment. It addresses many of the weaknesses of KDD-based datasets. However, it does not specifically target IoT network traffic patterns.

**Role in this project:** Considered as alternative baseline. Not selected as primary dataset due to lack of IoT specificity.

---

#### 3. CIC-IoT-2023 ✅ **Selected**

| Property | Value |
|----------|-------|
| Origin | Canadian Institute for Cybersecurity, 2023 |
| Size | ~46M records across 34 attack classes |
| Features | 46 raw → 33 after cleaning |
| Attack classes | 34 (DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce, ...) |
| Format | `.csv` per attack type |
| IoT specificity | High — generated from real IoT device testbed |

CIC-IoT-2023 was selected as the primary dataset for the following reasons:

- **Recency** — generated in 2023 from a real IoT testbed with 105 devices
- **Attack diversity** — 34 distinct attack classes covering modern IoT threat vectors
- **Scale** — sufficient volume for federated partitioning across 3 nodes
- **Realism** — traffic captured from physical IoT devices (cameras, thermostats, locks, sensors)
- **Research backing** — published in *Sensors* journal (DOI: 10.3390/s23135941)

---

## Dataset Download Instructions

### CIC-IoT-2023 (primary — required)

1. Visit the official page: https://www.unb.ca/cic/datasets/iotdataset-2023.html
2. Register and download the CSV files
3. Place files under the following structure:

```
experiments/fl-iot-ids-v1/data/raw/
├── node1/train.csv
├── node2/train.csv
└── node3/train.csv
```

The partitioning script (`src/scripts/prepare_partitions.py`) handles the split automatically from the full dataset. See the [fl-iot-ids-v1 README](../experiments/fl-iot-ids-v1/README.md) for full instructions.

### NSL-KDD (optional — QIEA experiments only)

1. Download from: https://www.unb.ca/cic/datasets/nsl.html
2. Place `.arff` files under:

```
data/NSL-KDD/nsl-kdd/
├── KDDTrain+.arff
├── KDDTrain+_20Percent.arff
├── KDDTest+.arff
└── KDDTest-21.arff
```

### UNSW-NB15 (optional — reference only)

1. Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Place CSV files under:

```
data/Federated-Learning-Based-Intrusion-Detection-System-main/data/
```

---

## QIEA Feature Selection — NSL-KDD

As part of the Quantum-Inspired module development (targeting v2), a preliminary experiment was conducted using the **Quantum-Inspired Evolutionary Algorithm (QIEA)** for feature selection on NSL-KDD.

### Objective

Instead of using all 41 features of NSL-KDD, the QIEA automatically selects the 10 most discriminative features for intrusion detection, reducing model complexity while preserving or improving classification performance.

### Method

The QIEA encodes each candidate feature subset as a quantum-inspired chromosome — a vector of probability amplitudes representing the likelihood of each feature being selected. At each generation, the algorithm applies quantum rotation gates to update amplitudes based on fitness, collapsing to binary feature masks for evaluation.

**Pipeline:**

1. Load and preprocess NSL-KDD (label encoding, standard scaling)
2. Initialize QIEA population with random quantum chromosomes (length = 41)
3. Evaluate each chromosome: train a `RandomForestClassifier` on the selected features, measure F1-macro on validation set
4. Apply quantum rotation update toward the best solution
5. Repair chromosomes to ensure exactly 10 features selected
6. Repeat for N generations, track convergence

### Result Summary

| Configuration | Features | Accuracy | Macro F1 |
|---|---|---|---|
| RandomForest — all 41 features | 41 | baseline | baseline |
| RandomForest — QIEA top-10 features | 10 | ≈ baseline | ≈ baseline |

> Full results and convergence plots are available in the experiment notebook: `experiments/baseline-CIC_IOT_2023/notebooks/CICIOT23_Pipeline.ipynb`

This experiment validates the QIEA concept before integrating it into the full federated pipeline in v2 (`fl-iot-ids-v2`).

---

## Directory Layout

```
data/
├── README.md                          ← this file
├── .gitkeep                           ← keeps directory tracked by Git
├── NSL-KDD/                           ← optional, QIEA experiments
│   └── nsl-kdd/
│       ├── KDDTrain+.arff
│       ├── KDDTest+.arff
│       └── ...
├── Federated-Learning-Based-Intrusion-Detection-System-main/
│   └── data/                          ← optional, UNSW-NB15 reference
│       └── *.csv
└── fl.pcap                            ← optional, raw capture sample
```

> All `.csv`, `.arff`, `.pcap`, `.npz`, and `.pkl` files are excluded from Git via `.gitignore`. Never commit raw datasets or preprocessed artifacts.

---

## References

- Nour, M. A., & Slay, J. (2015). *UNSW-NB15: A comprehensive data set for network intrusion detection systems.* MilCIS.
- Tavallaee, M. et al. (2009). *A detailed analysis of the KDD CUP 99 data set.* IEEE CISDA.
- Koroniotis, N. et al. (2023). *A new network intrusion detection dataset for IoT: CIC-IoT-2023.* Sensors, 23(13), 5941. https://doi.org/10.3390/s23135941