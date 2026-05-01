"""Tests for map_raw_to_features (edge-ids-gateway)."""
from __future__ import annotations

import copy

import pytest

from feature_mapper import CANONICAL_FEATURE_NAMES, map_raw_to_features

MINIMAL_EVENT = {
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

FULL_EVENT = {
    **MINIMAL_EVENT,
    "protocol_number": 6,
    "app_proto": "https",
    "ttl": 128,
    "header_bytes_total": 240,
    "min_packet_size": 60,
    "packet_size_std": 12.5,
    "iat_ns_mean": 14166666.0,
    "window_packet_mean": 6.0,
    "request_rate": 70.6,
}


def test_returns_exactly_28_features():
    features = map_raw_to_features(copy.deepcopy(MINIMAL_EVENT))
    assert len(features) == 28


def test_output_feature_names_match_canonical_in_order():
    features = map_raw_to_features(copy.deepcopy(MINIMAL_EVENT))
    assert list(features.keys()) == CANONICAL_FEATURE_NAMES


def test_flow_duration_maps_from_duration_ms():
    features = map_raw_to_features(copy.deepcopy(MINIMAL_EVENT))
    assert features["flow_duration"] == pytest.approx(85 / 1000.0)


def test_syn_flag_number_maps_from_flags_syn():
    features = map_raw_to_features(copy.deepcopy(MINIMAL_EVENT))
    assert features["syn_flag_number"] == 1.0


def test_tot_sum_equals_bytes_in_plus_bytes_out():
    features = map_raw_to_features(copy.deepcopy(MINIMAL_EVENT))
    assert features["Tot sum"] == pytest.approx(920 + 3980)


def test_minimal_event_all_optional_absent():
    features = map_raw_to_features(copy.deepcopy(MINIMAL_EVENT))
    assert len(features) == 28
    for value in features.values():
        assert isinstance(value, float)


def test_full_event_with_all_optional_fields():
    features = map_raw_to_features(copy.deepcopy(FULL_EVENT))
    assert len(features) == 28
    assert features["flow_duration"] == pytest.approx(85 / 1000.0)
    assert features["HTTPS"] == 1.0
    assert features["HTTP"] == 0.0
