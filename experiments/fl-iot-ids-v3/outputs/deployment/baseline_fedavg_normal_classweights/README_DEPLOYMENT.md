# FL Baseline Deployment Bundle

**Experiment:** `exp_v3_fedavg_normal_classweights`
**Strategy:** FedAvg | **Scenario:** normal_noniid | **Imbalance:** class_weights
**Best round:** 10 | **Macro-F1:** 0.6797 | **Benign recall:** 0.8792 | **FPR:** 0.1208

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
global_model.pth  : 393aa55b47dfecd6e1b95d3799c89f3a3ad40fb0e52eb63fead646cf54184f79
scaler.pkl        : aa0024644df57bcfaf5c953e46c5e1c35c821958113244685cd0a4df076028eb
feature_names.pkl : 22bf5e33a7778b923f94ec6663279847567fe85c02b743b6154bba51c20f0179
label_mapping.pkl : 51e0224f06e8a9c146e3f0d914982b646d71abd959f3aef384c0aac1d1165cc0
```

## Inference Snippet

```python
import pickle, json
import numpy as np
import torch
import torch.nn as nn

# --- Load bundle ---
BUNDLE = "outputs/deployment/baseline_fedavg_normal_classweights"

with open(f"{BUNDLE}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(f"{BUNDLE}/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open(f"{BUNDLE}/label_mapping.json") as f:
    label_mapping = json.load(f)

id_to_label = {int(k): v for k, v in label_mapping["id_to_label"].items()}

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
state_dict = torch.load(f"{BUNDLE}/global_model.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# --- Inference ---
sample = np.random.randn(1, 28).astype(np.float32)  # replace with real features
sample_scaled = scaler.transform(sample)
with torch.no_grad():
    logits = model(torch.tensor(sample_scaled))
    pred_id = int(logits.argmax(dim=1).item())

print(f"Predicted class: {id_to_label[pred_id]} (id={pred_id})")
```
