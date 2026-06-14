from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
EXTRACTOR_DIR = FINAL_ROOT / "deployment" / "live_feature_extractor"
sys.path.insert(0, str(EXTRACTOR_DIR))

from feature_schema import EXPECTED_FEATURES, load_schema, selection_decision_uses_test_holdout  # noqa: E402
from nfstream_adapter import dry_run_flow_stats  # noqa: E402
from pcap_to_features import DEFAULT_CSV_NAME, DEFAULT_GAP_REPORT_NAME, DEFAULT_JSON_NAME, project_mappings, run_extraction  # noqa: E402
from ciciot_feature_mapper import map_flows  # noqa: E402


def run_dry_validation() -> dict[str, object]:
    schema = load_schema()
    schema.validate()
    flows = dry_run_flow_stats()
    mappings = map_flows(flows, schema)
    selected_records, scaler_used = project_mappings(mappings, "selected_12_scaled", schema)

    with tempfile.TemporaryDirectory(prefix="p16_feature_extraction_") as tmp_dir:
        result = run_extraction(
            pcap_path=None,
            output_mode="selected_12_scaled",
            output_dir=tmp_dir,
            dry_run=True,
        )
        output_dir = Path(tmp_dir)
        outputs_exist = {
            DEFAULT_CSV_NAME: (output_dir / DEFAULT_CSV_NAME).exists(),
            DEFAULT_JSON_NAME: (output_dir / DEFAULT_JSON_NAME).exists(),
            DEFAULT_GAP_REPORT_NAME: (output_dir / DEFAULT_GAP_REPORT_NAME).exists(),
        }

    test_holdout_used = selection_decision_uses_test_holdout()
    checks = {
        "feature_order_is_28_expected_features": schema.feature_names == EXPECTED_FEATURES and len(schema.feature_names) == 28,
        "selected_12_scaled_returns_12_values": bool(selected_records) and len(selected_records[0].features) == 12,
        "selected_mask_id_is_conservative_seed_42": schema.selected_mask_id == "conservative_seed_42",
        "no_test_holdout_used_for_training_or_tuning": not test_holdout_used,
        "scaler_used_for_selected_12_scaled": scaler_used,
        "dry_outputs_written": all(outputs_exist.values()) and len(result.records) == 1,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "selected_mask_id": schema.selected_mask_id,
        "selected_indices": schema.selected_indices,
        "selected_feature_names": schema.selected_feature_names,
        "outputs_written_in_temp_dir": outputs_exist,
        "test_holdout_used_for_training_or_tuning": test_holdout_used,
    }


def main() -> int:
    result = run_dry_validation()
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

