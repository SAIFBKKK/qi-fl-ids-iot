from pathlib import Path

def create_partition_manifest(output_path: str | Path, partitions: dict) -> None:
    import json
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(partitions, f, indent=2)
