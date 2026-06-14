from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator


def load_replay_rows(path: str | Path) -> Iterator[dict[str, object]]:
    replay_path = Path(path)
    suffix = replay_path.suffix.lower()
    if suffix == ".jsonl":
        with replay_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value
        return

    if suffix == ".json":
        value = json.loads(replay_path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else value.get("rows", []) if isinstance(value, dict) else []
        for row in rows:
            if isinstance(row, dict):
                yield row
        return

    if suffix == ".csv":
        with replay_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield dict(row)
        return

    raise ValueError(f"unsupported replay file extension: {suffix}")


def features_from_row(row: dict[str, object]) -> list[float] | None:
    features = row.get("features")
    if isinstance(features, list):
        return [float(value) for value in features]
    if isinstance(features, str) and features.strip():
        parsed = json.loads(features)
        if isinstance(parsed, list):
            return [float(value) for value in parsed]
    return None

