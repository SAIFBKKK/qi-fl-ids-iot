from __future__ import annotations


VALID_TIERS = ("weak", "medium", "powerful")


def assign_tier(cpu_count: int | float, ram_gb: int | float) -> str:
    """Assign a live-lab tier from simple VM hardware constraints."""
    cpu = float(cpu_count)
    ram = float(ram_gb)
    if cpu <= 2 or ram <= 4:
        return "weak"
    if cpu <= 4 or ram <= 8:
        return "medium"
    return "powerful"

