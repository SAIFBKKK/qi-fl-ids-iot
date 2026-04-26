"""
Validate the baseline deployment bundle (10 tests).

Usage (from experiments/fl-iot-ids-v3/):
    python src/scripts/validate_bundle.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import sys
import textwrap
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BUNDLE_NAME = "baseline_fedavg_normal_classweights"
BUNDLE_DIR = ROOT / "outputs" / "deployment" / BUNDLE_NAME
LOGS_DIR = ROOT / "logs"

REQUIRED_FILES = [
    "global_model.pth",
    "scaler.pkl",
    "feature_names.pkl",
    "label_mapping.json",
    "label_mapping.pkl",
    "model_config.json",
    "run_summary.json",
    "README_DEPLOYMENT.md",
]

REQUIRED_MODEL_CONFIG_FIELDS = [
    "input_dim", "hidden_dims", "num_classes", "dropout",
    "macro_f1", "best_round", "sha256",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("validate_bundle")


def _add_file_handler() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOGS_DIR / "build_baseline.log", mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [VALIDATE] %(message)s"))
    logging.getLogger("validate_bundle").addHandler(fh)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_model(input_dim: int, num_classes: int, hidden_dims: list, dropout: float):
    import torch.nn as nn

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            h1, h2 = hidden_dims[0], hidden_dims[1]
            self.net = nn.Sequential(
                nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(h2, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    return _MLP()


class TestResult:
    def __init__(self, idx: int, name: str):
        self.idx = idx
        self.name = name
        self.passed = False
        self.message = ""

    def ok(self, msg: str = "") -> "TestResult":
        self.passed = True
        self.message = msg
        return self

    def fail(self, msg: str) -> "TestResult":
        self.passed = False
        self.message = msg
        return self

    def __str__(self) -> str:
        icon = "✓" if self.passed else "✗"
        status = "PASS" if self.passed else "FAIL"
        detail = f" — {self.message}" if self.message else ""
        return f"  [{icon}] Test {self.idx:02d}: {self.name} [{status}]{detail}"


def test_01_all_files_present() -> TestResult:
    r = TestResult(1, "Tous les 8 fichiers présents")
    missing = [f for f in REQUIRED_FILES if not (BUNDLE_DIR / f).exists()]
    if missing:
        return r.fail(f"Fichiers manquants: {missing}")
    return r.ok()


def test_02_model_loadable() -> TestResult:
    r = TestResult(2, "global_model.pth chargeable")
    try:
        state_dict = torch.load(BUNDLE_DIR / "global_model.pth", map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            return r.fail(f"Type inattendu: {type(state_dict)}")
        return r.ok(f"{len(state_dict)} clés")
    except Exception as e:
        return r.fail(str(e))


def test_03_state_dict_keys() -> TestResult:
    r = TestResult(3, "state_dict a exactement 6 clés attendues")
    expected_keys = {
        "net.0.weight", "net.0.bias",
        "net.3.weight", "net.3.bias",
        "net.6.weight", "net.6.bias",
    }
    try:
        state_dict = torch.load(BUNDLE_DIR / "global_model.pth", map_location="cpu", weights_only=True)
        actual_keys = set(state_dict.keys())
        if actual_keys != expected_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            return r.fail(f"missing={missing} extra={extra}")
        return r.ok(f"clés: {sorted(actual_keys)}")
    except Exception as e:
        return r.fail(str(e))


def test_04_scaler_transform() -> TestResult:
    r = TestResult(4, "scaler.pkl chargeable + transform retourne shape (1, 28)")
    try:
        with (BUNDLE_DIR / "scaler.pkl").open("rb") as f:
            scaler = pickle.load(f)
        sample = np.zeros((1, 28), dtype=np.float32)
        result = scaler.transform(sample)
        if result.shape != (1, 28):
            return r.fail(f"Shape inattendue: {result.shape}")
        return r.ok(f"shape={result.shape}")
    except Exception as e:
        return r.fail(str(e))


def test_05_feature_names_len() -> TestResult:
    r = TestResult(5, "feature_names.pkl → len == 28")
    try:
        with (BUNDLE_DIR / "feature_names.pkl").open("rb") as f:
            names = pickle.load(f)
        if len(names) != 28:
            return r.fail(f"len={len(names)} ≠ 28")
        return r.ok(f"len=28, premier='{names[0]}'")
    except Exception as e:
        return r.fail(str(e))


def test_06_label_mapping_consistency() -> TestResult:
    r = TestResult(6, "label_mapping.json == label_mapping.pkl (cohérence)")
    try:
        with (BUNDLE_DIR / "label_mapping.pkl").open("rb") as f:
            lm_pkl = pickle.load(f)
        with (BUNDLE_DIR / "label_mapping.json").open("r", encoding="utf-8") as f:
            lm_json = json.load(f)

        pkl_l2i = lm_pkl.get("label_to_id", {})
        json_l2i = lm_json.get("label_to_id", {})
        if pkl_l2i != json_l2i:
            diff = {k for k in pkl_l2i if pkl_l2i.get(k) != json_l2i.get(k)}
            return r.fail(f"label_to_id diverge sur: {list(diff)[:5]}")

        pkl_i2l = {str(k): v for k, v in lm_pkl.get("id_to_label", {}).items()}
        json_i2l = lm_json.get("id_to_label", {})
        if pkl_i2l != json_i2l:
            return r.fail("id_to_label diverge entre pkl et json")

        return r.ok("label_to_id et id_to_label cohérents")
    except Exception as e:
        return r.fail(str(e))


def test_07_label_mapping_34_entries() -> TestResult:
    r = TestResult(7, "label_mapping a exactement 34 entrées")
    try:
        with (BUNDLE_DIR / "label_mapping.json").open("r", encoding="utf-8") as f:
            lm = json.load(f)
        n = len(lm.get("label_to_id", {}))
        if n != 34:
            return r.fail(f"label_to_id a {n} entrées ≠ 34")
        n2 = len(lm.get("id_to_label", {}))
        if n2 != 34:
            return r.fail(f"id_to_label a {n2} entrées ≠ 34")
        return r.ok("34 classes confirmées")
    except Exception as e:
        return r.fail(str(e))


def test_08_model_config_fields() -> TestResult:
    r = TestResult(8, "model_config.json a tous les champs requis")
    try:
        with (BUNDLE_DIR / "model_config.json").open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        missing = [k for k in REQUIRED_MODEL_CONFIG_FIELDS if k not in cfg]
        if missing:
            return r.fail(f"Champs manquants: {missing}")
        sha = cfg.get("sha256", {})
        missing_sha = [k for k in ["global_model.pth", "scaler.pkl", "feature_names.pkl", "label_mapping.pkl"] if k not in sha]
        if missing_sha:
            return r.fail(f"sha256 manquants pour: {missing_sha}")
        return r.ok(f"macro_f1={cfg['macro_f1']:.4f} best_round={cfg['best_round']}")
    except Exception as e:
        return r.fail(str(e))


def test_09_end_to_end_inference() -> TestResult:
    r = TestResult(9, "Inférence end-to-end: sample aléatoire → classe valide [0, 33]")
    try:
        with (BUNDLE_DIR / "model_config.json").open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        with (BUNDLE_DIR / "scaler.pkl").open("rb") as f:
            scaler = pickle.load(f)

        state_dict = torch.load(BUNDLE_DIR / "global_model.pth", map_location="cpu", weights_only=True)
        model = _build_model(
            input_dim=cfg["input_dim"],
            num_classes=cfg["num_classes"],
            hidden_dims=cfg["hidden_dims"],
            dropout=cfg["dropout"],
        )
        model.load_state_dict(state_dict)
        model.eval()

        np.random.seed(0)
        sample = np.random.randn(1, cfg["input_dim"]).astype(np.float32)
        sample_scaled = scaler.transform(sample).astype(np.float32)

        with torch.no_grad():
            logits = model(torch.tensor(sample_scaled))
            pred_id = int(logits.argmax(dim=1).item())

        if not (0 <= pred_id < cfg["num_classes"]):
            return r.fail(f"pred_id={pred_id} hors [0, {cfg['num_classes'] - 1}]")

        with (BUNDLE_DIR / "label_mapping.json").open("r", encoding="utf-8") as f:
            lm = json.load(f)
        label = lm["id_to_label"].get(str(pred_id), "UNKNOWN")
        return r.ok(f"pred_id={pred_id} ({label})")
    except Exception as e:
        return r.fail(traceback.format_exc().splitlines()[-1])


def test_10_sha256_match() -> TestResult:
    r = TestResult(10, "sha256 dans model_config.json correspond aux fichiers réels")
    try:
        with (BUNDLE_DIR / "model_config.json").open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        expected = cfg.get("sha256", {})
        mismatches = []
        for fname, expected_hash in expected.items():
            fpath = BUNDLE_DIR / fname
            if not fpath.exists():
                mismatches.append(f"{fname}: fichier absent")
                continue
            actual_hash = _sha256_file(fpath)
            if actual_hash != expected_hash:
                mismatches.append(f"{fname}: attendu {expected_hash[:12]}… got {actual_hash[:12]}…")
        if mismatches:
            return r.fail("; ".join(mismatches))
        return r.ok(f"{len(expected)} fichiers vérifiés")
    except Exception as e:
        return r.fail(str(e))


ALL_TESTS = [
    test_01_all_files_present,
    test_02_model_loadable,
    test_03_state_dict_keys,
    test_04_scaler_transform,
    test_05_feature_names_len,
    test_06_label_mapping_consistency,
    test_07_label_mapping_34_entries,
    test_08_model_config_fields,
    test_09_end_to_end_inference,
    test_10_sha256_match,
]


def main() -> None:
    _add_file_handler()
    print(f"\n{'='*62}")
    print(f"  VALIDATION — {BUNDLE_NAME}")
    print(f"  {BUNDLE_DIR}")
    print(f"{'='*62}")

    results = [fn() for fn in ALL_TESTS]

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print()
    for r in results:
        print(str(r))
    print()
    print(f"{'='*62}")
    print(f"  Résultat : {passed}/{total} tests passés")
    print(f"{'='*62}\n")

    if passed < total:
        failed = [r.name for r in results if not r.passed]
        print(f"Tests échoués :\n" + "\n".join(f"  - {n}" for n in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
