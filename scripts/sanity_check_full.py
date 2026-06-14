#!/usr/bin/env python3
"""Full demo-readiness sanity check for QI-FL-IDS-IoT."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SERVICES_DIR = ROOT / "services"

FL_SERVER = os.getenv("SANITY_FL_SERVER_URL", "http://localhost:8080")
DASHBOARD = os.getenv("SANITY_DASHBOARD_URL", "http://localhost:8090")
PROMETHEUS = os.getenv("SANITY_PROMETHEUS_URL", "http://localhost:9090")
GRAFANA = os.getenv("SANITY_GRAFANA_URL", "http://localhost:3000")
STRICT_ASSIGNMENTS = os.getenv("SANITY_STRICT_ASSIGNMENTS", "0") == "1"

EXPECTED_DEFAULT_SERVICES = [
    "mosquitto",
    "traffic-generator",
    "mlflow",
    "fl-server",
    "dashboard",
    "prometheus",
    "grafana",
    "iot-node-1",
    "iot-node-2",
    "iot-node-3",
]


@dataclass
class HttpResult:
    ok: bool
    status: int
    body: str
    error: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class Category:
    key: str
    label: str
    ok: bool
    summary: str
    details: list[str] = field(default_factory=list)
    remediation: str = ""


def http_request(
    url: str,
    method: str = "GET",
    body: Any | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 5.0,
) -> HttpResult:
    data = None
    request_headers = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    req = Request(url, data=data, headers=request_headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as response:
            return HttpResult(
                ok=200 <= response.status < 400,
                status=response.status,
                body=response.read().decode("utf-8", errors="replace"),
                headers=dict(response.headers.items()),
            )
    except HTTPError as exc:
        return HttpResult(
            ok=False,
            status=exc.code,
            body=exc.read().decode("utf-8", errors="replace"),
            error=str(exc),
            headers=dict(exc.headers.items()) if exc.headers else {},
        )
    except URLError as exc:
        return HttpResult(ok=False, status=0, body="", error=str(exc.reason))
    except Exception as exc:  # noqa: BLE001 - diagnostic script must keep going.
        return HttpResult(ok=False, status=0, body="", error=str(exc))


def parse_json(result: HttpResult) -> Any:
    if not result.body:
        return None
    try:
        return json.loads(result.body)
    except json.JSONDecodeError:
        return None


def parse_compose_json(stdout: str) -> list[dict[str, Any]]:
    text = stdout.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows


def check_services() -> Category:
    try:
        proc = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            cwd=SERVICES_DIR,
            check=False,
            text=True,
            capture_output=True,
            timeout=12,
        )
    except Exception as exc:  # noqa: BLE001
        return Category(
            "A",
            "Services Docker",
            False,
            "docker compose ps unavailable",
            [str(exc)],
            "Lancer Docker Desktop puis docker compose up -d depuis services/.",
        )

    if proc.returncode != 0:
        return Category(
            "A",
            "Services Docker",
            False,
            "docker compose ps failed",
            [proc.stderr.strip() or proc.stdout.strip()],
            "Verifier que Docker Desktop tourne et que le contexte compose est accessible.",
        )

    rows = parse_compose_json(proc.stdout)
    by_service = {row.get("Service") or row.get("Name"): row for row in rows}
    missing = [name for name in EXPECTED_DEFAULT_SERVICES if name not in by_service]
    unhealthy = []
    for name in EXPECTED_DEFAULT_SERVICES:
        row = by_service.get(name)
        if not row:
            continue
        state = str(row.get("State", "")).lower()
        health = str(row.get("Health", "")).lower()
        status = str(row.get("Status", "")).lower()
        running = state == "running" or "up" in status
        healthy = health in {"", "healthy"} or "healthy" in status
        if not running or not healthy:
            unhealthy.append(f"{name}: state={state or '?'} health={health or '?'} status={status or '?'}")

    ok = not missing and not unhealthy
    details = []
    if missing:
        details.append("missing: " + ", ".join(missing))
    if unhealthy:
        details.extend(unhealthy)
    summary = f"{len(EXPECTED_DEFAULT_SERVICES) - len(missing) - len(unhealthy)}/{len(EXPECTED_DEFAULT_SERVICES)} running/healthy"
    return Category(
        "A",
        "Services Docker",
        ok,
        summary,
        details,
        "Relancer docker compose up -d et inspecter docker compose ps.",
    )


def check_fl_routes() -> Category:
    routes = ["/health", "/nodes", "/models", "/schedule"]
    details = []
    ok_count = 0
    for route in routes:
        result = http_request(f"{FL_SERVER}{route}")
        if result.ok:
            ok_count += 1
        else:
            details.append(f"{route}: HTTP {result.status} {result.error or result.body[:120]}")

    assignments = http_request(f"{FL_SERVER}/assignments")
    if assignments.ok:
        ok_count += 1
        summary = "5/5 routes"
        ok = len(details) == 0
    elif STRICT_ASSIGNMENTS:
        details.append(f"/assignments: HTTP {assignments.status} {assignments.error or assignments.body[:120]}")
        summary = f"{ok_count}/5 routes"
        ok = False
    else:
        details.append("/assignments: optional route unavailable (set SANITY_STRICT_ASSIGNMENTS=1 to fail)")
        summary = f"{ok_count}/4 required routes (+ optional /assignments unavailable)"
        ok = len(details) == 1

    return Category(
        "B",
        "fl-server routes",
        ok,
        summary,
        details,
        "Verifier fl-server logs; si necessaire implementer GET /assignments.",
    )


def check_md5() -> Category:
    result = http_request(f"{FL_SERVER}/models")
    data = parse_json(result) if result.ok else None
    tiers = (data or {}).get("tiers") or (data or {}).get("models") or []
    ready = [tier for tier in tiers if tier.get("ready") or tier.get("available")]
    md5_by_tier = {tier.get("tier"): tier.get("md5") for tier in ready if tier.get("md5")}
    required = ["weak", "medium", "powerful"]
    missing = [tier for tier in required if not md5_by_tier.get(tier)]
    md5s = [md5_by_tier[tier] for tier in required if md5_by_tier.get(tier)]
    distinct = len(md5s) == 3 and len(set(md5s)) == 3
    ok = result.ok and not missing and distinct
    detail = ", ".join(f"{tier}={md5_by_tier.get(tier, '-')}" for tier in required)
    return Category(
        "C",
        "3 MD5 distincts",
        ok,
        "weak != medium != powerful" if ok else "md5 check failed",
        [detail],
        "Verifier /models, model_factory_30rounds et le calcul md5 au boot.",
    )


def check_dashboard_tabs() -> Category:
    tabs = ["iot", "fl", "qi", "monitoring"]
    failures = []
    for tab in tabs:
        result = http_request(f"{DASHBOARD}/tab/{tab}")
        if not result.ok:
            failures.append(f"/tab/{tab}: HTTP {result.status} {result.error or result.body[:120]}")
    return Category(
        "D",
        "Dashboard tabs",
        not failures,
        f"{len(tabs) - len(failures)}/{len(tabs)} tabs OK",
        failures,
        "Rebuild dashboard puis verifier templates et logs uvicorn.",
    )


def check_dashboard_apis() -> Category:
    endpoints = [
        "/api/nodes",
        "/api/models",
        "/api/fl/health",
        "/api/fl/schedule",
        "/api/fl/runs?max_results=5",
        "/api/qi/overview",
        "/api/system/health",
        "/api/scenarios",
        "/api/prometheus/query?" + urlencode({"q": "registered_nodes_total"}),
    ]
    failures = []
    for endpoint in endpoints:
        result = http_request(f"{DASHBOARD}{endpoint}", timeout=8.0)
        data = parse_json(result) if result.ok else None
        if not result.ok or data is None:
            failures.append(f"{endpoint}: HTTP {result.status} {result.error or result.body[:120]}")
    return Category(
        "E",
        "Dashboard APIs",
        not failures,
        f"{len(endpoints) - len(failures)}/{len(endpoints)} routes",
        failures,
        "Verifier dashboard/api/*, FL_SERVER_URL, MLFLOW_URL et PROMETHEUS_URL.",
    )


def check_system_health() -> Category:
    result = http_request(f"{DASHBOARD}/api/system/health")
    data = parse_json(result) if result.ok else {}
    ok = result.ok and data.get("overall") == "healthy" and data.get("ups") == data.get("total") == 4
    services = data.get("services") or []
    details = [f"{svc.get('service')}: {svc.get('status')}" for svc in services]
    return Category(
        "F",
        "System health",
        ok,
        f"{data.get('overall', 'unknown')} {data.get('ups', '?')}/{data.get('total', '?')}",
        details,
        "Verifier /api/system/health et les URLs fl-server/mlflow/prometheus/grafana.",
    )


def check_scenarios() -> Category:
    result = http_request(f"{DASHBOARD}/api/scenarios")
    data = parse_json(result) if result.ok else {}
    scenarios = data.get("scenarios") or []
    ok = result.ok and len(scenarios) == 3
    names = [scenario.get("id", "?") for scenario in scenarios]
    return Category(
        "G",
        "Scenarios",
        ok,
        f"{len(scenarios)} disponibles",
        [", ".join(names)],
        "Verifier services/dashboard/scenarios et api/scenarios.py.",
    )


def check_grafana_uid() -> Category:
    result = http_request(f"{GRAFANA}/api/dashboards/uid/qi-fl-ids-overview", timeout=8.0)
    data = parse_json(result) if result.ok else {}
    title = ((data or {}).get("dashboard") or {}).get("title")
    page = http_request(f"{GRAFANA}/d/qi-fl-ids-overview/qi-fl-ids-overview?kiosk", timeout=8.0)
    ok = result.ok and title == "QI-FL-IDS Overview" and page.ok
    return Category(
        "H",
        "Grafana UID",
        ok,
        title or f"HTTP {result.status}",
        [f"dashboard API HTTP {result.status}", f"kiosk HTTP {page.status}"],
        "Restart grafana et verifier provisioning dashboards.",
    )


def check_cors_preflight() -> Category:
    result = http_request(
        f"{FL_SERVER}/nodes/register",
        method="OPTIONS",
        headers={
            "Origin": DASHBOARD,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    allow_origin = result.headers.get("access-control-allow-origin") or result.headers.get("Access-Control-Allow-Origin")
    allow_methods = result.headers.get("access-control-allow-methods") or result.headers.get("Access-Control-Allow-Methods")
    ok = result.ok and bool(allow_origin) and bool(allow_methods)
    return Category(
        "I",
        "CORS preflight",
        ok,
        "Access-Control headers OK" if ok else f"HTTP {result.status}",
        [f"allow-origin={allow_origin}", f"allow-methods={allow_methods}"],
        "Verifier CORSMiddleware fl-server allow_origins/methods/headers.",
    )


def print_report(categories: list[Category]) -> None:
    print("SANITY CHECK COMPLET - QI-FL-IDS-IoT")
    print("====================================")
    print()
    for category in categories:
        icon = "OK" if category.ok else "FAIL"
        print(f"[{category.key}] {category.label:<22} {icon:<5} {category.summary}")
        if not category.ok:
            for detail in category.details:
                if detail:
                    print(f"    - {detail}")
            if category.remediation:
                print(f"    remediation: {category.remediation}")
    print()
    print("====================================")
    ok_count = sum(1 for category in categories if category.ok)
    total = len(categories)
    if ok_count == total:
        print(f"OK {ok_count}/{total} categories - systeme pret pour demo")
    else:
        print(f"FAIL {ok_count}/{total} categories OK - correction requise avant demo")


def main() -> int:
    categories = [
        check_services(),
        check_fl_routes(),
        check_md5(),
        check_dashboard_tabs(),
        check_dashboard_apis(),
        check_system_health(),
        check_scenarios(),
        check_grafana_uid(),
        check_cors_preflight(),
    ]
    print_report(categories)
    return 0 if all(category.ok for category in categories) else 1


if __name__ == "__main__":
    sys.exit(main())
