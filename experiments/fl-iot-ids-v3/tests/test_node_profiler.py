from __future__ import annotations

import yaml
import pytest

from src.common.schemas import NodeProfile
from src.fl.node_profiler import NodeProfiler


def _write_tier_config(tmp_path):
    config_path = tmp_path / "tier_profiles.yaml"
    payload = {
        "tiers": {
            "weak": {
                "model_width": 0.25,
                "local_epochs": 1,
                "batch_size": 128,
                "max_ram_mb": 512,
            },
            "medium": {
                "model_width": 0.5,
                "local_epochs": 2,
                "batch_size": 64,
                "max_ram_mb": 2048,
            },
            "powerful": {
                "model_width": 1.0,
                "local_epochs": 3,
                "batch_size": 32,
                "max_ram_mb": None,
                "device_types": ["edge_pc", "server"],
            },
        }
    }
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)
    return config_path


def _profile(**overrides):
    payload = {
        "node_id": "nodeX",
        "cpu_cores": 2,
        "ram_mb": 1024,
        "device_type": "raspberry_pi_4",
        "avg_latency_ms": 50.0,
        "battery_powered": False,
        "network_quality": "medium",
    }
    payload.update(overrides)
    return NodeProfile.from_dict(payload)


def test_assigns_weak_for_low_ram(tmp_path):
    profiler = NodeProfiler(_write_tier_config(tmp_path))

    assignment = profiler.assign_tier(_profile(node_id="node1", ram_mb=256))

    assert assignment.assigned_tier == "weak"
    assert assignment.model_width == 0.25
    assert assignment.local_epochs == 1
    assert assignment.batch_size == 128


def test_assigns_medium_for_mid_ram(tmp_path):
    profiler = NodeProfiler(_write_tier_config(tmp_path))

    assignment = profiler.assign_tier(_profile(node_id="node2", ram_mb=1024))

    assert assignment.assigned_tier == "medium"
    assert assignment.model_width == 0.5


def test_assigns_powerful_for_edge_pc_with_high_ram(tmp_path):
    profiler = NodeProfiler(_write_tier_config(tmp_path))

    assignment = profiler.assign_tier(
        _profile(node_id="node3", ram_mb=4096, device_type="edge_pc")
    )

    assert assignment.assigned_tier == "powerful"
    assert assignment.model_width == 1.0


def test_profile_missing_fields_raises_value_error():
    with pytest.raises(ValueError, match="Missing NodeProfile fields"):
        NodeProfile.from_dict({"node_id": "node1", "ram_mb": 256})


def test_borderline_512_ram_is_medium(tmp_path):
    profiler = NodeProfiler(_write_tier_config(tmp_path))

    assignment = profiler.assign_tier(_profile(node_id="node2", ram_mb=512))

    assert assignment.assigned_tier == "medium"


def test_battery_powered_forces_weak_even_with_high_cpu_and_ram(tmp_path):
    profiler = NodeProfiler(_write_tier_config(tmp_path))

    assignment = profiler.assign_tier(
        _profile(
            node_id="node1",
            cpu_cores=16,
            ram_mb=8192,
            device_type="edge_pc",
            battery_powered=True,
        )
    )

    assert assignment.assigned_tier == "weak"


def test_yaml_loading_and_profile_serialization_round_trip(tmp_path):
    profiler = NodeProfiler(_write_tier_config(tmp_path))
    profile = _profile(node_id="node2", ram_mb=1024)

    loaded = NodeProfile.from_dict(profile.to_dict())
    assignment = profiler.assign_tier(loaded)

    assert loaded == profile
    assert assignment.to_dict() == {
        "assigned_tier": "medium",
        "model_width": 0.5,
        "local_epochs": 2,
        "batch_size": 64,
    }
    assert profiler.list_assignments() == {"node2": "medium"}
