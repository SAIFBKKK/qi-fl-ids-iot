from pathlib import Path

from src.utils.mlflow_logger import normalize_tracking_uri


def test_normalize_tracking_uri_converts_windows_absolute_paths():
    uri = normalize_tracking_uri(r"C:\Users\saifb\dev\qi-fl-ids-iot\outputs\mlruns")
    assert uri == "file:///C:/Users/saifb/dev/qi-fl-ids-iot/outputs/mlruns"


def test_normalize_tracking_uri_converts_posix_absolute_paths(tmp_path: Path):
    uri = normalize_tracking_uri(str(tmp_path / "mlruns"))
    assert uri == (tmp_path / "mlruns").resolve().as_uri()


def test_normalize_tracking_uri_leaves_remote_uris_unchanged():
    uri = "http://127.0.0.1:5000"
    assert normalize_tracking_uri(uri) == uri
