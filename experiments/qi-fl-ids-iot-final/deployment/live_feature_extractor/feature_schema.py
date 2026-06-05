from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_FEATURES = [
    "flow_duration",
    "Header_Length",
    "Protocol Type",
    "Duration",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "urg_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "SSH",
    "TCP",
    "UDP",
    "ARP",
    "ICMP",
    "Tot sum",
    "Min",
    "Std",
    "IAT",
    "Number",
]

DEFAULT_SELECTED_INDICES = [0, 2, 3, 4, 6, 13, 14, 19, 20, 25, 26, 27]
DEFAULT_SELECTED_MASK_ID = "conservative_seed_42"
INPUT_MODES = ("original_28_unscaled", "original_28_scaled", "selected_12_scaled")


@dataclass(frozen=True)
class FeatureSchema:
    feature_names: list[str]
    selected_indices: list[int]
    selected_mask_id: str

    @property
    def selected_feature_names(self) -> list[str]:
        return [self.feature_names[index] for index in self.selected_indices]

    def validate(self) -> None:
        if self.feature_names != EXPECTED_FEATURES:
            raise ValueError("feature order does not match outputs/artifacts/features/feature_names.json contract")
        if len(self.selected_indices) != 12:
            raise ValueError(f"expected 12 selected features, got {len(self.selected_indices)}")
        if self.selected_mask_id != DEFAULT_SELECTED_MASK_ID:
            raise ValueError(f"unexpected selected_mask_id: {self.selected_mask_id}")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def final_root() -> Path:
    return repo_root() / "experiments" / "qi-fl-ids-iot-final"


def default_feature_names_path() -> Path:
    return final_root() / "outputs" / "artifacts" / "features" / "feature_names.json"


def default_deployment_schema_path() -> Path:
    return final_root() / "deployment" / "l1_final" / "feature_schema.json"


def default_qga_mask_path() -> Path:
    return final_root() / "outputs" / "qga_feature_selection" / "final_selected_mask" / "feature_mask.json"


def default_selection_decision_path() -> Path:
    return final_root() / "outputs" / "qga_feature_selection" / "final_selected_mask" / "selection_decision.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_names(path: str | Path | None = None) -> list[str]:
    feature_path = Path(path) if path else default_feature_names_path()
    if not feature_path.exists():
        return list(EXPECTED_FEATURES)
    value = read_json(feature_path)
    if not isinstance(value, list):
        raise ValueError(f"feature_names must be a list: {feature_path}")
    return [str(item) for item in value]


def load_selected_indices(schema_path: str | Path | None = None, mask_path: str | Path | None = None) -> list[int]:
    deployment_schema_path = Path(schema_path) if schema_path else default_deployment_schema_path()
    if deployment_schema_path.exists():
        schema = read_json(deployment_schema_path)
        indices = schema.get("selected_indices")
        if indices:
            return [int(item) for item in indices]

    qga_mask_path = Path(mask_path) if mask_path else default_qga_mask_path()
    if qga_mask_path.exists():
        mask = read_json(qga_mask_path)
        indices = mask.get("selected_indices")
        if indices:
            return [int(item) for item in indices]

    return list(DEFAULT_SELECTED_INDICES)


def load_selected_mask_id(mask_path: str | Path | None = None) -> str:
    qga_mask_path = Path(mask_path) if mask_path else default_qga_mask_path()
    if not qga_mask_path.exists():
        return DEFAULT_SELECTED_MASK_ID
    mask = read_json(qga_mask_path)
    return str(mask.get("mask_id", DEFAULT_SELECTED_MASK_ID))


def load_schema() -> FeatureSchema:
    schema = FeatureSchema(
        feature_names=load_feature_names(),
        selected_indices=load_selected_indices(),
        selected_mask_id=load_selected_mask_id(),
    )
    schema.validate()
    return schema


def selected_from_original(values: list[float | None], indices: list[int] | None = None) -> list[float | None]:
    selected_indices = indices or load_selected_indices()
    return [values[index] for index in selected_indices]


def selection_decision_uses_test_holdout() -> bool:
    path = default_selection_decision_path()
    if not path.exists():
        return False
    decision = read_json(path)
    return bool(decision.get("test_used_for_selection", False))


def schema_summary() -> dict[str, object]:
    schema = load_schema()
    return {
        "feature_count": len(schema.feature_names),
        "selected_feature_count": len(schema.selected_indices),
        "feature_names": schema.feature_names,
        "selected_indices": schema.selected_indices,
        "selected_mask_id": schema.selected_mask_id,
        "input_modes": list(INPUT_MODES),
        "test_holdout_used_for_training_or_tuning": selection_decision_uses_test_holdout(),
    }

