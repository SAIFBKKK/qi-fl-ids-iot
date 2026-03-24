from pathlib import Path

class LocalCollector:
    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)

    def list_files(self):
        return sorted(self.raw_dir.glob("*"))

    def exists(self) -> bool:
        return self.raw_dir.exists()
