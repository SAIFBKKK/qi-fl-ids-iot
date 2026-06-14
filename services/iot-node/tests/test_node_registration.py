import sys
from types import SimpleNamespace

from node_registration import HardwareProfile, collect_hardware_profile, register_with_model_server


def test_fallback_without_model_server_url():
    profile = HardwareProfile(
        node_id="node1",
        cpu_cores=2,
        ram_mb=1024,
        device_type="docker_node",
        network_quality="medium",
        battery_powered=False,
    )

    state = register_with_model_server(None, profile)

    assert state.assigned_tier == "legacy"
    assert state.model_source == "local_artifacts"
    assert state.status == "local_fallback"


def test_registration_with_model_server_mock(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "node_id": "node1",
                "assigned_tier": "weak",
                "model_version": "placeholder",
                "model_source": "local_registry",
                "status": "registered",
            }

    calls = {}

    def fake_post(url, json, timeout):
        calls["url"] = url
        calls["json"] = json
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=fake_post))
    profile = HardwareProfile(
        node_id="node1",
        cpu_cores=2,
        ram_mb=1024,
        device_type="docker_node",
        network_quality="medium",
        battery_powered=False,
        tier_override="weak",
    )

    state = register_with_model_server("http://fl-server:8080", profile)

    assert calls["url"] == "http://fl-server:8080/nodes/register"
    assert calls["json"]["tier_override"] == "weak"
    assert state.assigned_tier == "weak"
    assert state.status == "registered"


def test_collect_hardware_profile_uses_env(monkeypatch):
    monkeypatch.setenv("NODE_ID", "node1")
    monkeypatch.setenv("RAM_MB", "1024")
    monkeypatch.setenv("DEVICE_TYPE", "docker_node")
    monkeypatch.setenv("NETWORK_QUALITY", "medium")
    monkeypatch.setenv("BATTERY_POWERED", "false")
    monkeypatch.setenv("NODE_TIER_OVERRIDE", "weak")

    profile = collect_hardware_profile()

    assert profile.node_id == "node1"
    assert profile.ram_mb == 1024
    assert profile.tier_override == "weak"
