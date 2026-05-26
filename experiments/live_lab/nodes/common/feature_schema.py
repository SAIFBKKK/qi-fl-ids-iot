from __future__ import annotations

import json
from pathlib import Path

from utils import repo_root_from_live_lab


FALLBACK_FEATURES = [
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


def feature_names_path() -> Path:
    return repo_root_from_live_lab() / "experiments" / "qi-fl-ids-iot-final" / "outputs" / "artifacts" / "features" / "feature_names.json"


def load_feature_names(path: str | Path | None = None) -> list[str]:
    target = Path(path) if path else feature_names_path()
    if not target.exists():
        return list(FALLBACK_FEATURES)
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"feature names file is not a list: {target}")
    return [str(item) for item in value]


def expected_28_features() -> list[str]:
    names = load_feature_names()
    if len(names) != 28:
        raise ValueError(f"expected 28 features, got {len(names)}")
    return names

