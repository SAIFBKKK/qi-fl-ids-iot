from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"
REPORT_DIR = FINAL_ROOT / "outputs" / "reports"

NODES = ("iot-rpi-weak", "iot-smart-watch-medium")
QUERY_SPECS = {
    "final_mqtt_bridge_metrics": (8016, "/metrics"),
    "online_validator_summary": (8015, "/summary"),
    "final_ids_api_metrics": (8014, "/metrics"),
    "live_lab_controller_nodes": (8020, "/nodes"),
    "live_lab_controller_assignments": (8020, "/assignments"),
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def build_url(server_url: str, port: int, path: str) -> str:
    candidate = server_url if "://" in server_url else f"http://{server_url}"
    parsed = urllib.parse.urlparse(candidate)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or parsed.path
    return urllib.parse.urlunparse((scheme, f"{host}:{port}", path, "", "", ""))


def fetch_url(url: str, timeout: float) -> dict[str, Any]:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "p16-8-window-evidence/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read().decode("utf-8", errors="replace")
            return {
                "ok": True,
                "url": url,
                "status": getattr(response, "status", None),
                "bytes": len(content.encode("utf-8")),
                "content": content,
            }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "url": url, "warning": str(exc)}


def fetch_all(server_url: str, timeout: float) -> dict[str, dict[str, Any]]:
    return {name: fetch_url(build_url(server_url, port, path), timeout) for name, (port, path) in QUERY_SPECS.items()}


def prometheus_metric_sum(text: str, metric_name: str, labels: dict[str, str] | None = None) -> float | None:
    labels = labels or {}
    total = 0.0
    found = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith(metric_name):
            continue
        if labels and not all(f'{key}="{value}"' in line for key, value in labels.items()):
            continue
        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            total += float(parts[1])
            found = True
        except ValueError:
            continue
    return total if found else None


def parse_json(content: str | None) -> Any:
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def parse_topic_counts(content: str | None) -> dict[str, Any]:
    payload = parse_json(content)
    if not isinstance(payload, dict):
        return {}
    counts = payload.get("topic_counts", {})
    return counts if isinstance(counts, dict) else {}


def extract_evidence(fetches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bridge_metrics = fetches.get("final_mqtt_bridge_metrics", {}).get("content", "")
    api_metrics = fetches.get("final_ids_api_metrics", {}).get("content", "")
    summary = fetches.get("online_validator_summary", {}).get("content")
    topic_counts = parse_topic_counts(summary)
    node_evidence: dict[str, Any] = {}
    for node_id in NODES:
        node_evidence[node_id] = {
            "flows_received_total": prometheus_metric_sum(
                bridge_metrics, "final_mqtt_bridge_flows_received_total", {"node_id": node_id}
            ),
            "predictions_published_total": prometheus_metric_sum(
                bridge_metrics, "final_mqtt_bridge_predictions_published_total", {"node_id": node_id}
            ),
            "alerts_published_total": prometheus_metric_sum(
                bridge_metrics, "final_mqtt_bridge_alerts_published_total", {"node_id": node_id}
            ),
            "validator_flow_topic_count": topic_counts.get(f"ids/flows/{node_id}"),
            "validator_prediction_topic_count": topic_counts.get(f"ids/predictions/{node_id}"),
            "validator_alert_topic_count": topic_counts.get(f"ids/alerts/{node_id}"),
        }
    return {
        "nodes": node_evidence,
        "errors": {
            "final_ids_api_prediction_errors_total": prometheus_metric_sum(
                api_metrics, "final_ids_api_prediction_errors_total"
            ),
            "final_mqtt_bridge_prediction_errors_total": prometheus_metric_sum(
                bridge_metrics, "final_mqtt_bridge_prediction_errors_total"
            ),
        },
        "controller_nodes": parse_json(fetches.get("live_lab_controller_nodes", {}).get("content")),
        "controller_assignments": parse_json(fetches.get("live_lab_controller_assignments", {}).get("content")),
    }


def sanitize_fetches(fetches: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: {key: value for key, value in result.items() if key != "content"} for name, result in fetches.items()}


def build_report(server_url: str, fetches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    warnings = [
        f"{name}: {result.get('warning')}"
        for name, result in fetches.items()
        if not result.get("ok")
    ]
    return {
        "generated_at": utc_now(),
        "server_url": server_url,
        "scope": "P16.8 controlled SyntheticPacketSource window publication evidence",
        "note": "Run this collector after controlled P16.8 publishes for final runtime evidence; otherwise counters reflect the current server state.",
        "live_fetches": sanitize_fetches(fetches),
        "evidence": extract_evidence(fetches),
        "warnings": warnings,
    }


def value_or_na(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "p16_8_window_publish_evidence.json"
    md_path = REPORT_DIR / "p16_8_window_publish_evidence_table.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# P16.8 Window Publish Evidence Table",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Server URL: `{report['server_url']}`",
        "- Scope: controlled SyntheticPacketSource packet windows only.",
        "- Note: run this collector after controlled P16.8 publishes for final runtime evidence.",
        "",
        "## Node Evidence",
        "",
        "| Node | Bridge flows | Bridge predictions | Bridge alerts | Validator flows | Validator predictions | Validator alerts |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for node_id in NODES:
        node = report["evidence"]["nodes"][node_id]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{node_id}`",
                    value_or_na(node.get("flows_received_total")),
                    value_or_na(node.get("predictions_published_total")),
                    value_or_na(node.get("alerts_published_total")),
                    value_or_na(node.get("validator_flow_topic_count")),
                    value_or_na(node.get("validator_prediction_topic_count")),
                    value_or_na(node.get("validator_alert_topic_count")),
                ]
            )
            + " |"
        )
    errors = report["evidence"]["errors"]
    lines.extend(
        [
            "",
            "## Error Counters",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| `final_ids_api_prediction_errors_total` | {value_or_na(errors.get('final_ids_api_prediction_errors_total'))} |",
            f"| `final_mqtt_bridge_prediction_errors_total` | {value_or_na(errors.get('final_mqtt_bridge_prediction_errors_total'))} |",
            "",
            "## Warnings",
            "",
        ]
    )
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- None.")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect P16.8 controlled window publish evidence.")
    parser.add_argument("--server-url", default="http://192.168.56.1")
    parser.add_argument("--timeout", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.server_url, fetch_all(args.server_url, args.timeout))
    write_reports(report)
    print(
        json.dumps(
            {
                "ok": True,
                "warnings": report["warnings"],
                "json": str(REPORT_DIR / "p16_8_window_publish_evidence.json"),
                "markdown": str(REPORT_DIR / "p16_8_window_publish_evidence_table.md"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
