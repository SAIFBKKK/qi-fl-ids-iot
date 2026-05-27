from __future__ import annotations


VALID_TIERS = ("weak", "medium", "powerful")


def assign_tier(cpu_count: int | float, ram_gb: int | float, device_type: str | None = None) -> str:
    """Assign a live-lab tier from declared device role, then resources."""
    normalized_device_type = (device_type or "").strip().lower()
    if normalized_device_type == "raspberry_like":
        return "weak"
    if normalized_device_type == "smart_watch_like":
        return "medium"

    cpu = float(cpu_count)
    ram = float(ram_gb)
    if cpu <= 2 or ram <= 4:
        return "weak"
    if cpu <= 4 or ram <= 8:
        return "medium"
    return "powerful"
