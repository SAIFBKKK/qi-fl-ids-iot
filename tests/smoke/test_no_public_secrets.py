import subprocess
from pathlib import Path


FORBIDDEN_TRACKED_NAMES = {
    ".claude/settings.local.json",
    ".vscode/settings.json",
    "services/.env",
    "services/mosquitto/passwords",
    "kaggle.json",
    "data/fl.pcap",
}

RISK_PATTERNS = [
    ".env",
    "kaggle.json",
    "passwords",
    ".pem",
    ".key",
]


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]
    return [path for path in paths if Path(path).exists()]


def test_known_private_files_are_not_tracked() -> None:
    tracked = set(tracked_files())
    offenders = sorted(FORBIDDEN_TRACKED_NAMES & tracked)
    assert not offenders, f"Private files are tracked: {offenders}"


def test_no_obvious_tracked_secret_filenames() -> None:
    offenders = []
    for path in tracked_files():
        lower = path.lower()
        if lower.endswith(".env.example"):
            continue
        if "generate_mqtt_password.sh" in lower:
            continue
        if any(pattern in lower for pattern in RISK_PATTERNS):
            offenders.append(path)

    assert not offenders, "Tracked secret-risk filenames: " + ", ".join(offenders[:50])
