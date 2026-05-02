from tier_assignment import NodeProfile, assign_tier


def test_assigns_weak_for_small_cpu_or_ram():
    assert assign_tier(NodeProfile(node_id="n1", cpu_cores=2, ram_mb=8192)) == "weak"
    assert assign_tier(NodeProfile(node_id="n2", cpu_cores=8, ram_mb=1024)) == "weak"


def test_assigns_medium_for_mid_profile():
    assert assign_tier(NodeProfile(node_id="n1", cpu_cores=4, ram_mb=8192)) == "medium"
    assert assign_tier(NodeProfile(node_id="n2", cpu_cores=8, ram_mb=3072)) == "medium"


def test_assigns_powerful_for_large_profile():
    assert assign_tier(NodeProfile(node_id="n1", cpu_cores=8, ram_mb=8192)) == "powerful"


def test_tier_override_is_respected():
    profile = NodeProfile(node_id="n1", cpu_cores=1, ram_mb=512, tier_override="powerful")
    assert assign_tier(profile) == "powerful"
