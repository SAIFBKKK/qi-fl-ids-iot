from __future__ import annotations

import argparse
import json


def build_server_connection_check(server_url: str) -> dict[str, str]:
    return {"server_url": server_url.rstrip("/"), "mode": "dry_run", "status": "not_contacted"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run server connection check.")
    parser.add_argument("--server-url", default="http://SERVER_IP:8020")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_server_connection_check(args.server_url), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

