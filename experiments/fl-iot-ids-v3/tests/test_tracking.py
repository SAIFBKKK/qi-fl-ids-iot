from pathlib import Path

from src.tracking.artifact_logger import BaselineArtifactTracker, build_mlflow_round_metrics
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


def test_tracker_exports_round_curves_for_mlflow():
    tracker = BaselineArtifactTracker(
        experiment={"name": "exp_test"},
        config={"strategy": {"num_rounds": 3}},
    )
    tracker.record_fit_round(
        1,
        {
            "train_loss_last": 0.42,
            "train_time_sec": 1.5,
            "update_size_bytes": 2048,
        },
    )
    tracker.record_evaluate_round(
        1,
        distributed_loss=0.33,
        metrics={
            "accuracy": 0.91,
            "macro_f1": 0.88,
            "rare_class_recall": 0.77,
        },
    )

    series = tracker.build_mlflow_round_series()

    assert series == [
        (
            1,
            {
                "train/loss": 0.42,
                "train/time_sec": 1.5,
                "communication/update_size_bytes": 2048.0,
                "validation/loss": 0.33,
                "validation/accuracy": 0.91,
                "validation/macro_f1": 0.88,
                "validation/rare_class_recall": 0.77,
            },
        )
    ]


def test_build_mlflow_round_metrics_supports_live_round_logging():
    metrics = build_mlflow_round_metrics(
        {
            "train_loss_last": 0.5,
            "accuracy": 0.9,
        },
        distributed_loss=0.25,
    )

    assert metrics == {
        "train/loss": 0.5,
        "validation/loss": 0.25,
        "validation/accuracy": 0.9,
    }
