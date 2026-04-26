"""
Build the deployment bundle for the FL baseline experiment.

Usage (from experiments/fl-iot-ids-v3/):
    python src/scripts/build_baseline_bundle.py

Produces:
    outputs/deployment/baseline_fedavg_normal_classweights/
    ├── global_model.pth
    ├── scaler.pkl
    ├── feature_names.pkl
    ├── label_mapping.json
    ├── label_mapping.pkl
    ├── model_config.json
    ├── run_summary.json
    └── README_DEPLOYMENT.md
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXPERIMENT_NAME = "exp_v3_fedavg_normal_classweights"
BUNDLE_NAME = "baseline_fedavg_normal_classweights"

CHECKPOINT_PATH = ROOT / "outputs" / "reports" / "baselines" / EXPERIMENT_NAME / "best_checkpoint.pth"
RUN_SUMMARY_PATH = ROOT / "outputs" / "reports" / "baselines" / EXPERIMENT_NAME / "run_summary.json"
RESOLVED_CONFIG_PATH = ROOT / "outputs" / "reports" / "baselines" / EXPERIMENT_NAME / "resolved_config.json"
ARTIFACTS_DIR = ROOT / "artifacts"
BUNDLE_DIR = ROOT / "outputs" / "deployment" / BUNDLE_NAME
LOGS_DIR = ROOT / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("build_baseline_bundle")


def _add_file_handler() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOGS_DIR / "build_baseline.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        logger.error("best_checkpoint.pth not found at %s", CHECKPOINT_PATH)
        logger.error(
            "Run: python src/scripts/run_experiment.py --experiment %s", EXPERIMENT_NAME
        )
        sys.exit(1)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        logger.error("Checkpoint does not contain 'state_dict' key. Keys: %s", list(checkpoint.keys()))
        sys.exit(1)
    logger.info(
        "Loaded checkpoint: round=%s macro_f1=%.4f saved_at=%s",
        checkpoint.get("round"),
        checkpoint.get("macro_f1") or 0.0,
        checkpoint.get("saved_at"),
    )
    return checkpoint


def copy_global_model(checkpoint: dict) -> Path:
    dst = BUNDLE_DIR / "global_model.pth"
    torch.save(checkpoint["state_dict"], dst)
    logger.info("global_model.pth written (%d bytes)", dst.stat().st_size)
    return dst


def copy_scaler() -> Path:
    src = ARTIFACTS_DIR / "scaler_standard_train_normal_noniid.pkl"
    if not src.exists():
        logger.error("scaler not found: %s", src)
        sys.exit(1)
    dst = BUNDLE_DIR / "scaler.pkl"
    shutil.copy2(src, dst)
    logger.info("scaler.pkl copied")
    return dst


def copy_feature_names() -> Path:
    src = ARTIFACTS_DIR / "feature_names_normal_noniid.pkl"
    if not src.exists():
        logger.error("feature_names not found: %s", src)
        sys.exit(1)
    dst = BUNDLE_DIR / "feature_names.pkl"
    shutil.copy2(src, dst)
    logger.info("feature_names.pkl copied")
    return dst


def copy_label_mapping() -> tuple[Path, Path]:
    src = ARTIFACTS_DIR / "baseline" / "artifacts" / "label_mapping_34.pkl"
    if not src.exists():
        logger.error("label_mapping not found: %s", src)
        sys.exit(1)

    dst_pkl = BUNDLE_DIR / "label_mapping.pkl"
    shutil.copy2(src, dst_pkl)

    with src.open("rb") as f:
        lm = pickle.load(f)

    label_to_id = lm.get("label_to_id", {})
    id_to_label = lm.get("id_to_label", {})
    json_payload = {
        "label_to_id": label_to_id,
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
    }
    dst_json = BUNDLE_DIR / "label_mapping.json"
    with dst_json.open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    logger.info("label_mapping.pkl + label_mapping.json written (%d classes)", len(label_to_id))
    return dst_pkl, dst_json


def build_model_config(checkpoint: dict, sha256s: dict[str, str]) -> Path:
    with RESOLVED_CONFIG_PATH.open("r", encoding="utf-8") as f:
        resolved = json.load(f)

    run_summary: dict = {}
    if RUN_SUMMARY_PATH.exists():
        with RUN_SUMMARY_PATH.open("r", encoding="utf-8") as f:
            run_summary = json.load(f)

    model_cfg = resolved.get("config", {}).get("model", {})
    best_round_info = run_summary.get("best_round", {})

    payload = {
        "architecture": resolved.get("experiment", {}).get("architecture", "flat_34"),
        "input_dim": int(model_cfg.get("input_dim", 28)),
        "hidden_dims": model_cfg.get("hidden_dims", [256, 128]),
        "num_classes": int(model_cfg.get("output_dim", 34)),
        "dropout": float(model_cfg.get("dropout", 0.2)),
        "best_round": int(best_round_info.get("best_round", checkpoint.get("round", 0))),
        "macro_f1": float(run_summary.get("final_macro_f1") or checkpoint.get("macro_f1") or 0.0),
        "benign_recall": float(
            run_summary.get("final_benign_recall") or checkpoint.get("benign_recall") or 0.0
        ),
        "false_positive_rate": float(
            run_summary.get("final_false_positive_rate")
            or checkpoint.get("false_positive_rate")
            or 0.0
        ),
        "experiment": EXPERIMENT_NAME,
        "fl_strategy": resolved.get("experiment", {}).get("fl_strategy", "fedavg"),
        "data_scenario": resolved.get("experiment", {}).get("data_scenario", "normal_noniid"),
        "imbalance_strategy": resolved.get("experiment", {}).get("imbalance_strategy", "class_weights"),
        "seed": int(resolved.get("config", {}).get("project", {}).get("seed", 42)),
        "torch_version": checkpoint.get("torch_version", torch.__version__),
        "checkpoint_saved_at": checkpoint.get("saved_at"),
        "bundle_built_at": datetime.utcnow().isoformat(),
        "sha256": sha256s,
    }

    dst = BUNDLE_DIR / "model_config.json"
    with dst.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("model_config.json written")
    return dst


def copy_run_summary() -> Path:
    dst = BUNDLE_DIR / "run_summary.json"
    shutil.copy2(RUN_SUMMARY_PATH, dst)
    logger.info("run_summary.json copied")
    return dst


def build_readme(model_cfg: dict) -> Path:
    macro_f1 = model_cfg.get("macro_f1", 0.0)
    benign_recall = model_cfg.get("benign_recall", 0.0)
    fpr = model_cfg.get("false_positive_rate", 0.0)
    best_round = model_cfg.get("best_round", "?")
    sha256s = model_cfg.get("sha256", {})

    content = f"""# FL Baseline Deployment Bundle

**Experiment:** `{EXPERIMENT_NAME}`
**Strategy:** FedAvg | **Scenario:** normal_noniid | **Imbalance:** class_weights
**Best round:** {best_round} | **Macro-F1:** {macro_f1:.4f} | **Benign recall:** {benign_recall:.4f} | **FPR:** {fpr:.4f}

## Files

| File | Description |
|------|-------------|
| `global_model.pth` | PyTorch state_dict of the best global model |
| `scaler.pkl` | StandardScaler fitted on normal_noniid train split |
| `feature_names.pkl` | List of 28 feature names |
| `label_mapping.json` | label↔id mapping (34 classes) |
| `label_mapping.pkl` | Same mapping as pickle |
| `model_config.json` | Architecture + metrics + SHA-256 manifest |
| `run_summary.json` | Full FL run summary |

## SHA-256 Manifest

```
global_model.pth  : {sha256s.get("global_model.pth", "N/A")}
scaler.pkl        : {sha256s.get("scaler.pkl", "N/A")}
feature_names.pkl : {sha256s.get("feature_names.pkl", "N/A")}
label_mapping.pkl : {sha256s.get("label_mapping.pkl", "N/A")}
```

## Inference Snippet

```python
import pickle, json
import numpy as np
import torch
import torch.nn as nn

# --- Load bundle ---
BUNDLE = "outputs/deployment/{BUNDLE_NAME}"

with open(f"{{BUNDLE}}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(f"{{BUNDLE}}/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open(f"{{BUNDLE}}/label_mapping.json") as f:
    label_mapping = json.load(f)

id_to_label = {{int(k): v for k, v in label_mapping["id_to_label"].items()}}

# --- Rebuild model ---
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(256, 128), dropout=0.2):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )
    def forward(self, x):
        return self.net(x)

model = MLPClassifier(input_dim=28, num_classes=34, hidden_dims=(256, 128), dropout=0.2)
state_dict = torch.load(f"{{BUNDLE}}/global_model.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# --- Inference ---
sample = np.random.randn(1, 28).astype(np.float32)  # replace with real features
sample_scaled = scaler.transform(sample)
with torch.no_grad():
    logits = model(torch.tensor(sample_scaled))
    pred_id = int(logits.argmax(dim=1).item())

print(f"Predicted class: {{id_to_label[pred_id]}} (id={{pred_id}})")
```
"""
    dst = BUNDLE_DIR / "README_DEPLOYMENT.md"
    with dst.open("w", encoding="utf-8") as f:
        f.write(content)
    logger.info("README_DEPLOYMENT.md written")
    return dst


def main() -> None:
    _add_file_handler()
    logger.info("=" * 60)
    logger.info("Building baseline bundle: %s", BUNDLE_NAME)
    logger.info("=" * 60)

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint()

    model_path = copy_global_model(checkpoint)
    scaler_path = copy_scaler()
    feature_names_path = copy_feature_names()
    label_mapping_pkl_path, _ = copy_label_mapping()
    copy_run_summary()

    sha256s = {
        "global_model.pth": sha256_file(model_path),
        "scaler.pkl": sha256_file(scaler_path),
        "feature_names.pkl": sha256_file(feature_names_path),
        "label_mapping.pkl": sha256_file(label_mapping_pkl_path),
    }
    logger.info("SHA-256 computed for 4 binary files")

    model_cfg_path = build_model_config(checkpoint, sha256s)
    with model_cfg_path.open("r", encoding="utf-8") as f:
        model_cfg = json.load(f)

    build_readme(model_cfg)

    total_size = sum(p.stat().st_size for p in BUNDLE_DIR.iterdir() if p.is_file())
    files = sorted(p.name for p in BUNDLE_DIR.iterdir() if p.is_file())

    logger.info("=" * 60)
    logger.info("Bundle complete: %s", BUNDLE_DIR)
    logger.info("Files (%d): %s", len(files), ", ".join(files))
    logger.info("Total size: %.1f KB", total_size / 1024)
    logger.info("macro_f1=%.4f  benign_recall=%.4f  FPR=%.4f",
                model_cfg["macro_f1"], model_cfg["benign_recall"], model_cfg["false_positive_rate"])
    logger.info("=" * 60)

    if total_size > 1_000_000:
        logger.warning("Bundle exceeds 1 MB (%.1f KB) — check for unexpectedly large files", total_size / 1024)


if __name__ == "__main__":
    main()
