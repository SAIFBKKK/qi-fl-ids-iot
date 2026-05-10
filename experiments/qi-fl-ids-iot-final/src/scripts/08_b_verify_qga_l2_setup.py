"""Verify P8-b QGA L2 setup without training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qga_l2.config import load_config, repo_path, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    checks = {}
    for key in ["l2_train_npz", "l2_val_npz", "l2_test_npz", "l2_family_mapping", "l2_partitions_root", "feature_names"]:
        path = repo_path(config, f"inputs.{key}")
        checks[key] = path.exists()
    scenario = repo_path(config, "inputs.l2_partitions_root") / "alpha_0.5" / "k3"
    checks["alpha_0.5_k3_exists"] = scenario.exists()
    checks["global_test_not_partitioned"] = not any((scenario / f"client_{i}" / "test_scaled.npz").exists() for i in range(1, 4))
    accepted = all(checks.values())
    payload = {
        "accepted": accepted,
        "phase": "P8-b",
        "mode": "verify",
        "checks": checks,
        "test_used_for_selection": False,
        "true_flower_runtime_required_for_final_fl": True,
        "warnings": [],
        "errors": [] if accepted else [key for key, value in checks.items() if not value],
    }
    write_json(repo_path(config, "outputs.reports_dir") / "p8b_qga_l2_verify_summary.json", payload)
    print(f"P8-b QGA L2 verify | accepted={accepted}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
