"""Collect P15 endpoint evidence from the local deployment stack."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


FINAL_DIR = Path("experiments/qi-fl-ids-iot-final")
DEFAULT_REPORTS_DIR = FINAL_DIR / "outputs" / "reports"


DEFAULT_ENDPOINTS = {
    "final_ids_api_health": "http://127.0.0.1:8014/health",
    "final_ids_api_ready": "http://127.0.0.1:8014/ready",
    "final_ids_api_metrics": "http://127.0.0.1:8014/metrics",
    "dashboard_p13_health": "http://127.0.0.1:8013/health",
    "traffic_generator_health": "http://127.0.0.1:8010/health",
    "online_validator_health": "http://127.0.0.1:8015/health",
    "online_validator_ready": "http://127.0.0.1:8015/ready",
    "online_validator_metrics": "http://127.0.0.1:8015/metrics",
}


def fetch(url: str, timeout_sec: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            status_code = response.status
            raw = response.read().decode("utf-8", errors="replace")
        latency_ms = (time.perf_counter() - started) * 1000.0
        parsed: Any
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw[:1000]
        return {
            "url": url,
            "reachable": True,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 3),
            "body": parsed,
            "error": None,
        }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "url": url,
            "reachable": False,
            "status_code": None,
            "latency_ms": round(latency_ms, 3),
            "body": None,
            "error": str(exc),
        }


def write_table(path: Path, results: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# P15 Online Evidence",
        "",
        "| Endpoint | Reachable | Status | Latency ms | Error |",
        "|---|---:|---:|---:|---|",
    ]
    for name, row in results.items():
        lines.append(
            f"| {name} | {row.get('reachable')} | {row.get('status_code')} | "
            f"{row.get('latency_ms')} | {row.get('error') or ''} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect health/readiness/metrics evidence from P14/P15 services.")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--timeout-sec", type=float, default=3.0)
    args = parser.parse_args()

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    results = {name: fetch(url, args.timeout_sec) for name, url in DEFAULT_ENDPOINTS.items()}
    summary = {
        "phase": "P15",
        "mode": "online_evidence_collection",
        "generated_at_unix": time.time(),
        "endpoints": results,
        "reachable_count": sum(1 for row in results.values() if row["reachable"]),
        "total_endpoints": len(results),
        "accepted": results["final_ids_api_health"]["reachable"] and results["dashboard_p13_health"]["reachable"],
    }
    (args.reports_dir / "p15_online_evidence.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_table(args.reports_dir / "p15_online_evidence_table.md", results)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
