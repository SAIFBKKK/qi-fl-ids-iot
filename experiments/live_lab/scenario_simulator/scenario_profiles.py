from __future__ import annotations


SCENARIOS = {
    "benign": {"description": "controlled baseline feature replay"},
    "ddos_dos_like": {"description": "controlled high-rate-like feature replay"},
    "recon_like": {"description": "controlled probing-like feature replay"},
    "web_based_like": {"description": "controlled web-like feature replay"},
    "brute_force_like": {"description": "controlled repeated-auth-like feature replay"},
    "spoofing_like": {"description": "controlled identity-change-like feature replay"},
    "mirai_like": {"description": "controlled botnet-family-like feature replay"},
}


def scenario_names() -> list[str]:
    return sorted(SCENARIOS)

