from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_FEATURES = [
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_feature_names_path() -> Path:
    return repo_root() / "experiments" / "qi-fl-ids-iot-final" / "outputs" / "artifacts" / "features" / "feature_names.json"


def default_deployment_schema_path() -> Path:
    return repo_root() / "experiments" / "qi-fl-ids-iot-final" / "deployment" / "l1_final" / "feature_schema.json"


def load_feature_names(path: str | Path | None = None) -> list[str]:
    feature_path = Path(path) if path else default_feature_names_path()
    if not feature_path.exists():
        return list(DEFAULT_FEATURES)
    value = json.loads(feature_path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"feature_names must be a list: {feature_path}")
    return [str(item) for item in value]


def load_selected_indices(path: str | Path | None = None) -> list[int]:
    schema_path = Path(path) if path else default_deployment_schema_path()
    if not schema_path.exists():
        return list(DEFAULT_SELECTED_INDICES)
    value = json.loads(schema_path.read_text(encoding="utf-8"))
    indices = value.get("selected_indices", DEFAULT_SELECTED_INDICES)
    return [int(item) for item in indices]


def align_feature_row(raw_features: dict[str, Any], feature_names: list[str] | None = None) -> tuple[list[float | None], list[str]]:
    names = feature_names or load_feature_names()
    values: list[float | None] = []
    unsupported: list[str] = []
    for name in names:
        value = raw_features.get(name)
        if value is None:
            values.append(None)
            unsupported.append(name)
            continue
        values.append(float(value))
    return values, unsupported


def selected_from_original(values: list[float | None], indices: list[int] | None = None) -> list[float | None]:
    selected_indices = indices or load_selected_indices()
    return [values[index] for index in selected_indices]


def schema_summary() -> dict[str, object]:
    feature_names = load_feature_names()
    selected_indices = load_selected_indices()
    return {
        "feature_count": len(feature_names),
        "selected_feature_count": len(selected_indices),
        "feature_names": feature_names,
        "selected_indices": selected_indices,
        "input_modes": ["selected_12_scaled", "original_28_scaled"],
    }

