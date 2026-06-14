from node_registry import NodeRegistry


def test_node_registration_returns_expected_payload_fields():
    registry = NodeRegistry()
    node = registry.register(
        {
            "node_id": "iot-node-1",
            "cpu_cores": 2,
            "ram_mb": 1024,
            "device_type": "docker_node",
            "network_quality": "medium",
            "battery_powered": False,
        }
    )

    assert node.node_id == "iot-node-1"
    assert node.assigned_tier == "weak"
    assert node.model_version == "placeholder"
    assert node.model_source == "local_registry"
    assert node.status == "registered"
    assert registry.total() == 1
    assert registry.counts_by_tier()["weak"] == 1
