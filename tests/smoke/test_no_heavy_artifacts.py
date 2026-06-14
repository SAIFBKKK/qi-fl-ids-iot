import subprocess
from pathlib import Path


HEAVY_SUFFIXES = {
    ".npz",
    ".npy",
    ".pth",
    ".pt",
    ".pkl",
    ".pickle",
    ".joblib",
    ".parquet",
    ".csv",
    ".pcap",
    ".zip",
}

ALLOWED_TRACKED_HEAVY = {
    "data/cic-iot-2023/demo_subsets/ddos_burst.parquet",
    "data/cic-iot-2023/demo_subsets/dos_slow.parquet",
    "data/cic-iot-2023/demo_subsets/mirai_wave.parquet",
    "data/cic-iot-2023/demo_subsets/mixed_chaos.parquet",
    "data/cic-iot-2023/demo_subsets/normal_traffic.parquet",
    "data/cic-iot-2023/demo_subsets/recon_scan.parquet",
}


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]
    return [path for path in paths if Path(path).exists()]


def test_no_unapproved_tracked_heavy_artifacts() -> None:
    offenders = []
    for path in tracked_files():
        suffix = Path(path).suffix.lower()
        if suffix in HEAVY_SUFFIXES and path not in ALLOWED_TRACKED_HEAVY:
            offenders.append(path)

    assert not offenders, "Unexpected tracked heavy artifacts: " + ", ".join(offenders[:50])


def test_packet_capture_is_not_tracked() -> None:
    assert "data/fl.pcap" not in tracked_files()
