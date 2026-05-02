#!/usr/bin/env python3
"""Sanity check du backend fl-server après Sprint 0."""
import json
import sys
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

BASE = "http://localhost:8080"
PROM = "http://localhost:8000"

def check(name, url, method="GET", body=None, expect_status=200):
    try:
        req = Request(url, method=method)
        if body:
            req.add_header("Content-Type", "application/json")
            req.data = json.dumps(body).encode()
        with urlopen(req, timeout=5) as r:
            data = r.read().decode()
            status = r.status
            ok = status == expect_status
            return ok, status, data
    except HTTPError as e:
        return False, e.code, str(e)
    except URLError as e:
        return False, 0, str(e)

results = []

# A - health
ok, status, _ = check("health", f"{BASE}/health")
results.append(("A", "fl-server /health", ok, status))

# B - swagger
ok, status, _ = check("docs", f"{BASE}/docs")
results.append(("B", "Swagger /docs", ok, status))

# C - 3 tiers distincts
ok, status, body = check("models", f"{BASE}/models")
md5s_distinct = False
if ok:
    try:
        data = json.loads(body)
        tiers = data.get("tiers", [])
        md5s = [t.get("md5") for t in tiers if t.get("available")]
        md5s_distinct = len(md5s) == 3 and len(set(md5s)) == 3
    except Exception:
        pass
results.append(("C", "3 MD5 distincts (KEY)", md5s_distinct, "OK" if md5s_distinct else "FAIL"))

# D - nodes
ok, status, body = check("nodes", f"{BASE}/nodes")
node_count = 0
if ok:
    try:
        node_count = len(json.loads(body).get("nodes", []))
    except Exception:
        pass
results.append(("D", f"GET /nodes ({node_count} nodes)", ok, status))

# E - register
ok, status, body = check("register", f"{BASE}/nodes/register", method="POST", body={
    "node_id": "sanity_test",
    "cpu_cores": 4,
    "ram_mb": 2048,
    "device_type": "test"
})
results.append(("E", "POST /nodes/register", ok, status))

# F - assignments
ok, status, _ = check("assignments", f"{BASE}/assignments")
results.append(("F", "GET /assignments", ok, status))

# G - CORS preflight
import urllib.request
try:
    req = urllib.request.Request(f"{BASE}/nodes", method="OPTIONS")
    req.add_header("Origin", "http://localhost:8090")
    req.add_header("Access-Control-Request-Method", "GET")
    with urlopen(req, timeout=5) as r:
        cors_ok = "Access-Control-Allow-Origin" in r.headers
    results.append(("G", "CORS preflight", cors_ok, "headers OK" if cors_ok else "no CORS"))
except Exception as e:
    results.append(("G", "CORS preflight", False, str(e)))

# H - Prometheus
ok, status, body = check("prom", f"{PROM}/metrics")
has_fl_metrics = "fl_current_round" in body if ok else False
results.append(("H", "Prometheus port 8000", has_fl_metrics, "fl_* found" if has_fl_metrics else "no fl_*"))

# Affichage
print()
print(f"{'#':<3} {'Test':<35} {'OK':<5} {'Detail'}")
print("-" * 70)
for letter, name, ok, detail in results:
    icon = "✓" if ok else "✗"
    print(f"{letter:<3} {name:<35} {icon:<5} {detail}")

print()
failures = [r for r in results if not r[2]]
if failures:
    print(f"❌ {len(failures)} test(s) failed:")
    for letter, name, _, _ in failures:
        print(f"   - {letter}: {name}")
    sys.exit(1)
else:
    print("✅ All sanity checks passed.")
    sys.exit(0)