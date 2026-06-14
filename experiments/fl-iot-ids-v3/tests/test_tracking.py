from pathlib import Path

from src.tracking.artifact_logger import BaselineArtifactTracker, build_mlflow_round_metrics
from src.utils.mlflow_logger import normalize_tracking_uri


def _tracker(fl_strategy: str = "qifa") -> BaselineArtifactTracker:
    return BaselineArtifactTracker(
        experiment={
            "name": f"exp_{fl_strategy}_test",
            "architecture": "flat_34",
            "fl_strategy": fl_strategy,
            "data_scenario": "normal_noniid",
            "imbalance_strategy": "class_weights",
        },
        config={
            "strategy": {"num_rounds": 1},
            "scenario": {"num_clients": 3},
            "dataset": {"feature_count": 28, "num_classes": 34},
            "project": {"seed": 42},
        },
    )


def _core_fit_metrics() -> dict[str, float]:
    return {
        "train_loss_last": 0.42,
        "train_time_sec": 1.5,
        "update_size_bytes": 2048,
    }


def _core_evaluate_metrics() -> dict[str, float]:
    return {
        "accuracy": 0.91,
        "macro_f1": 0.88,
        "recall_macro": 0.87,
        "benign_recall": 0.86,
        "false_positive_rate": 0.14,
        "rare_class_recall": 0.77,
        "rare_macro_f1": 0.70,
    }


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


def test_tracker_completion_does_not_require_strategy_optional_metrics():
    tracker = _tracker("qifa")
    tracker.record_fit_round(
        1,
        {
            **_core_fit_metrics(),
            "qifa_lambda": 0.15,
        },
    )
    tracker.record_evaluate_round(
        1,
        distributed_loss=0.33,
        metrics=_core_evaluate_metrics(),
    )

    summary = tracker.build_run_summary(status="success", duration_sec=1.0)

    assert summary["status"] == "success"
    assert summary["completed_rounds"] == 1


def test_tracker_completion_does_not_require_qifa_guard_optional_metrics():
    tracker = _tracker("qifa_guard")
    tracker.record_fit_round(1, _core_fit_metrics())
    tracker.record_evaluate_round(
        1,
        distributed_loss=0.33,
        metrics=_core_evaluate_metrics(),
    )

    summary = tracker.build_run_summary(status="success", duration_sec=1.0)

    assert summary["status"] == "success"
    assert summary["completed_rounds"] == 1


def test_tracker_marks_round_partial_when_core_metric_missing():
    tracker = _tracker("qifa")
    fit_metrics = _core_fit_metrics()
    fit_metrics.pop("update_size_bytes")
    tracker.record_fit_round(1, fit_metrics)
    tracker.record_evaluate_round(
        1,
        distributed_loss=0.33,
        metrics=_core_evaluate_metrics(),
    )

    summary = tracker.build_run_summary(status="success", duration_sec=1.0)

    assert summary["status"] == "partial"
    assert summary["completed_rounds"] == 0


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
