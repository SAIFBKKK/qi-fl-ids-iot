"""Tests for validate_raw_event (edge-ids-gateway)."""
from __future__ import annotations

import copy

import pytest

from raw_schema import validate_raw_event

VALID_EVENT = {
    "schema_version": "1.0",
    "event_type": "raw_iot_event",
    "timestamp": "2026-01-01T00:00:00Z",
    "node_id": "sensor-a1",
    "gateway_id": "node1",
    "node_group": "room-a",
    "device_type": "thermostat",
    "src_ip": "10.10.1.21",
    "dst_ip": "10.10.0.10",
    "src_port": 51544,
    "dst_port": 443,
    "protocol": "tcp",
    "packet_size": 820,
    "packet_count": 6,
    "duration_ms": 85,
    "bytes_in": 920,
    "bytes_out": 3980,
    "flags": {"syn": 1},
    "flag_counts": {},
    "scenario": "normal_traffic",
}


def test_valid_payload_passes():
    result = validate_raw_event(copy.deepcopy(VALID_EVENT))
    assert result["schema_version"] == "1.0"
    assert result["event_type"] == "raw_iot_event"
    assert result["protocol"] == "tcp"


def test_missing_src_ip_fails():
    payload = copy.deepcopy(VALID_EVENT)
    del payload["src_ip"]
    with pytest.raises(ValueError, match="src_ip"):
        validate_raw_event(payload)


def test_missing_protocol_fails():
    payload = copy.deepcopy(VALID_EVENT)
    del payload["protocol"]
    with pytest.raises(ValueError, match="protocol"):
        validate_raw_event(payload)


def test_invalid_ip_address_format_fails():
    payload = copy.deepcopy(VALID_EVENT)
    payload["src_ip"] = "not-an-ip-address"
    with pytest.raises(ValueError, match="src_ip"):
        validate_raw_event(payload)


def test_packet_count_zero_rejected():
    payload = copy.deepcopy(VALID_EVENT)
    payload["packet_count"] = 0
    with pytest.raises(ValueError, match="packet_count"):
        validate_raw_event(payload)
