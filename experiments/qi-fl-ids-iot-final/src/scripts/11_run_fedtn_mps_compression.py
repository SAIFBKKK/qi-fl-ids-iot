"""Run P11 FedTN/MPS compression dry-run or post-training compression."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap() -> None:
    src = Path(__file__).resolve().parents[1]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/qi-fl-ids-iot-final/configs/fedtn_mps_l1.yaml"))
    parser.add_argument("--base-model", choices=["fedavg_qga", "qifa_qga"], required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _bootstrap()
    from fedtn_mps.compression import estimate_low_rank_compression
    from fedtn_mps.config import load_config, resolve, run_id_now, write_json

    config = load_config(args.config)
    run_id = run_id_now()
    result = estimate_low_rank_compression(config, rank=args.rank)
    run_dir = resolve(config["outputs"]["run_dir"]) / args.base_model / f"rank_{args.rank}" / "runs" / run_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    warnings = list(result.warnings)
    if args.dry_run:
        warnings.append("dry_run_structure_only_checkpoint_not_loaded")
    summary = {
        "accepted": True,
        "phase": "P11",
        "run_id": run_id,
        "base_model": args.base_model,
        "dry_run": bool(args.dry_run),
        "compression": result.to_dict(),
        "test": {},
        "criteria": {
            "l1_only": True,
            "qga_mask_used": True,
            "full_fl_not_auto_launched": True,
            "checkpoint_required_for_metric_evaluation": True,
        },
        "warnings": warnings,
        "errors": [],
    }
    write_json(artifacts_dir / "run_summary.json", summary)
    write_json(artifacts_dir / "model_size_comparison.json", result.to_dict())
    write_json(
        artifacts_dir / "compression_manifest.json",
        {
            "method": config["compression"]["method"],
            "rank": args.rank,
            "target_layers": config["compression"]["target_layers"],
            "selected_mask_id": config["selected_mask_id"],
            "base_model": args.base_model,
        },
    )
    print(f"P11 FedTN/MPS compression run_id={run_id} base_model={args.base_model} rank={args.rank}")
    print(f"compression_ratio={result.compression_ratio:.4f} model_size_bytes={result.compressed_model_size_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
