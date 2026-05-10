"""Label and schema helpers for P1 data validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPECTED_FEATURES: list[str] = [
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

LABEL_COLUMN = "label_id"
EXPECTED_COLUMNS: list[str] = [*EXPECTED_FEATURES, LABEL_COLUMN]
EXPECTED_NUM_CLASSES = 34
BENIGN_LABEL_NAME = "BenignTraffic"
BENIGN_LABEL_ID = 1


def load_label_mapping(path: Path) -> dict[str, int]:
    """Load the canonical label -> id mapping."""
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("label_mapping.json must contain a JSON object")

    mapping: dict[str, int] = {}
    for label, value in raw.items():
        if not isinstance(label, str):
            raise ValueError("all label names must be strings")
        if not isinstance(value, int):
            raise ValueError(f"label id for {label!r} must be an integer")
        mapping[label] = value
    return mapping


def build_id_to_label(label_mapping: dict[str, int]) -> dict[int, str]:
    """Invert a label -> id mapping, rejecting duplicate ids."""
    id_to_label: dict[int, str] = {}
    for label, label_id in label_mapping.items():
        if label_id in id_to_label:
            raise ValueError(
                f"duplicate label id {label_id}: {id_to_label[label_id]!r} and {label!r}"
            )
        id_to_label[label_id] = label
    return dict(sorted(id_to_label.items()))


def validate_label_mapping(label_mapping: dict[str, int]) -> list[str]:
    """Return critical validation errors for the canonical mapping."""
    errors: list[str] = []
    ids = sorted(label_mapping.values())
    expected_ids = list(range(EXPECTED_NUM_CLASSES))

    if len(label_mapping) != EXPECTED_NUM_CLASSES:
        errors.append(
            f"expected {EXPECTED_NUM_CLASSES} label classes, found {len(label_mapping)}"
        )
    if ids != expected_ids:
        errors.append(f"expected label ids 0..33, found {ids}")
    if BENIGN_LABEL_NAME not in label_mapping:
        errors.append(f"{BENIGN_LABEL_NAME} is missing from label mapping")
    elif label_mapping[BENIGN_LABEL_NAME] != BENIGN_LABEL_ID:
        errors.append(
            f"{BENIGN_LABEL_NAME} must be id {BENIGN_LABEL_ID}, "
            f"found {label_mapping[BENIGN_LABEL_NAME]}"
        )
    return errors


def family_for_label(label_name: str) -> str:
    """Map a 34-class CIC-IoT label to the agreed L2 family."""
    if label_name == "BenignTraffic":
        return "Benign"
    if label_name.startswith("DDoS-"):
        return "DDoS"
    if label_name.startswith("DoS-"):
        return "DoS"
    if label_name.startswith("Recon-") or label_name == "VulnerabilityScan":
        return "Recon"
    if label_name in {
        "XSS",
        "SqlInjection",
        "CommandInjection",
        "BrowserHijacking",
        "Uploading_Attack",
    }:
        return "Web-based"
    if label_name == "DictionaryBruteForce":
        return "BruteForce"
    if label_name in {"DNS_Spoofing", "MITM-ArpSpoofing"}:
        return "Spoofing"
    if label_name.startswith("Mirai-"):
        return "Mirai"
    if label_name == "Backdoor_Malware":
        return "Malware"
    raise ValueError(f"no L2 family rule for label {label_name!r}")


def build_label_to_family(label_mapping: dict[str, int]) -> dict[str, str]:
    """Build the label -> L2 family mapping."""
    return {
        label: family_for_label(label)
        for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])
    }


def build_label_to_binary(label_mapping: dict[str, int]) -> dict[str, dict[str, Any]]:
    """Build the label -> L1 binary mapping.

    The original class id 1 (BenignTraffic) maps to binary label 0 for L1.
    All other original classes map to binary label 1.
    """
    binary: dict[str, dict[str, Any]] = {}
    for label, label_id in sorted(label_mapping.items(), key=lambda item: item[1]):
        is_benign = label == BENIGN_LABEL_NAME
        binary[label] = {
            "label_id": label_id,
            "binary_label": 0 if is_benign else 1,
            "binary_name": "normal" if is_benign else "attack",
        }
    return binary


def json_ready_id_to_label(id_to_label: dict[int, str]) -> dict[str, str]:
    """Convert integer JSON keys to strings deterministically."""
    return {str(label_id): label for label_id, label in sorted(id_to_label.items())}
