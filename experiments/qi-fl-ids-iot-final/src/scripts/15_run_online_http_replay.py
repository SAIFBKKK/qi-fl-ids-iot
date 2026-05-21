"""P15 HTTP replay against the final IDS API.

The script replays held-out L1 rows to ``final-ids-api /predict`` and records
online latency and IDS metrics. It never trains, updates, selects, or tunes the
model; the test holdout is only used as deployment replay evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


FINAL_DIR = Path("experiments/qi-fl-ids-iot-final")
DEFAULT_REPORTS_DIR = FINAL_DIR / "outputs" / "reports"
DEFAULT_TEST_NPZ = FINAL_DIR / "outputs" / "preprocessed" / "l1_binary" / "test_scaled.npz"
DEFAULT_FEATURE_SCHEMA = FINAL_DIR / "deployment" / "l1_final" / "feature_schema.json"
DEFAULT_DEPLOYMENT_15 = Path(
    "experiments/fl-iot-ids-v3/outputs/model_factory_30rounds/deployment_data/deployment_15.parquet"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_schema(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Feature schema not found: {path}")
    schema = read_json(path)
    selected_indices = [int(value) for value in schema.get("selected_indices", [])]
    selected_count = int(schema.get("selected_feature_count", len(selected_indices) or 12))
    if len(selected_indices) != selected_count:
        raise ValueError(f"Feature schema selected_indices mismatch: {len(selected_indices)} != {selected_count}")
    return schema


def label_array_from_npz(data: np.lib.npyio.NpzFile, path: Path) -> np.ndarray | None:
    for key in ("y", "y_binary", "labels", "label", "target"):
        if key in data:
            return data[key].astype(int)
    print(f"[WARN] No labels found in {path}; metrics will be latency-only.")
    return None


def load_test_npz(path: Path) -> tuple[np.ndarray, np.ndarray | None, list[Any]]:
    data = np.load(path, allow_pickle=True)
    if "X" not in data:
        raise KeyError(f"Missing X array in {path}")
    X = data["X"].astype(np.float32)
    y = label_array_from_npz(data, path)
    row_ids = data["row_id"].tolist() if "row_id" in data else list(range(len(X)))
    return X, y, row_ids


def load_deployment_15(path: Path, schema: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None, list[Any]]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on local optional dependency
        raise RuntimeError("pandas is required for --source deployment_15") from exc

    if not path.exists():
        raise FileNotFoundError(f"deployment_15 parquet not found: {path}")

    frame = pd.read_parquet(path)
    feature_names = schema.get("all_features", [])
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"deployment_15 missing feature columns: {missing[:5]}")

    X = frame[feature_names].to_numpy(dtype=np.float32)
    y: np.ndarray | None = None
    for key in ("y_binary", "label_binary", "binary_label"):
        if key in frame.columns:
            y = frame[key].to_numpy(dtype=int)
            break
    if y is None and "label_id" in frame.columns:
        y = (frame["label_id"].to_numpy(dtype=int) != 0).astype(int)

    if "row_id" in frame.columns:
        row_ids = frame["row_id"].tolist()
    elif "flow_id" in frame.columns:
        row_ids = frame["flow_id"].tolist()
    else:
        row_ids = list(range(len(X)))

    return X, y, row_ids


def select_rows(
    X: np.ndarray,
    y: np.ndarray | None,
    row_ids: list[Any],
    max_rows: int,
    use_qga_mask: bool,
    schema: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None, list[Any]]:
    limit = min(int(max_rows), len(X)) if max_rows > 0 else len(X)
    X = X[:limit]
    y = y[:limit] if y is not None else None
    row_ids = row_ids[:limit]

    if use_qga_mask:
        indices = [int(value) for value in schema.get("selected_indices", [])]
        X = X[:, indices]
    return X, y, row_ids


def post_predict(api_url: str, features: list[float], timeout_sec: float) -> tuple[dict[str, Any] | None, str | None, float]:
    url = f"{api_url.rstrip('/')}/predict"
    payload = json.dumps({"features": features}, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8")
        latency_ms = (time.perf_counter() - started) * 1000.0
        return json.loads(body), None, latency_ms
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return None, str(exc), latency_ms


def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    total = max(tp + tn + fp + fn, 1)

    precision_attack = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_attack = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_attack = (
        2 * precision_attack * recall_attack / (precision_attack + recall_attack)
        if (precision_attack + recall_attack) > 0
        else 0.0
    )

    precision_normal = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_normal = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_normal = (
        2 * precision_normal * recall_normal / (precision_normal + recall_normal)
        if (precision_normal + recall_normal) > 0
        else 0.0
    )

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": (tp + tn) / total,
        "attack_recall": recall_attack,
        "fpr": fpr,
        "fnr": fnr,
        "macro_f1": (f1_normal + f1_attack) / 2.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(values)
    p95_idx = min(int(round(0.95 * (len(ordered) - 1))), len(ordered) - 1)
    return {
        "mean_ms": float(statistics.mean(values)),
        "median_ms": float(statistics.median(values)),
        "p95_ms": float(ordered[p95_idx]),
        "max_ms": float(max(values)),
    }


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "row_index",
        "row_id",
        "label_true",
        "prediction",
        "probability_attack",
        "label_pred",
        "latency_ms",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary.get("online_metrics", {})
    latency = summary.get("latency", {})
    lines = [
        "# P15 Online HTTP Replay Summary",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| source | {summary.get('source')} |",
        f"| dry_run | {summary.get('dry_run')} |",
        f"| use_qga_mask | {summary.get('use_qga_mask')} |",
        f"| rows_attempted | {summary.get('rows_attempted')} |",
        f"| rows_predicted | {summary.get('rows_predicted')} |",
        f"| errors | {summary.get('errors')} |",
        f"| mean_latency_ms | {latency.get('mean_ms')} |",
        f"| p95_latency_ms | {latency.get('p95_ms')} |",
        f"| online_accuracy | {metrics.get('accuracy')} |",
        f"| online_attack_recall | {metrics.get('attack_recall')} |",
        f"| online_fpr | {metrics.get('fpr')} |",
        f"| online_fnr | {metrics.get('fnr')} |",
        f"| TP | {metrics.get('tp')} |",
        f"| TN | {metrics.get('tn')} |",
        f"| FP | {metrics.get('fp')} |",
        f"| FN | {metrics.get('fn')} |",
        "",
        "The held-out rows are used only for final online replay evidence. They are not used for training, mask selection, threshold tuning, or hyperparameter selection.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay L1 holdout rows to final-ids-api /predict.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8014")
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--sleep-ms", type=float, default=0.0)
    parser.add_argument("--use-qga-mask", action="store_true")
    parser.add_argument("--source", choices=["test_scaled_npz", "deployment_15"], default="test_scaled_npz")
    parser.add_argument("--test-npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--deployment-15", type=Path, default=DEFAULT_DEPLOYMENT_15)
    parser.add_argument("--feature-schema", type=Path, default=DEFAULT_FEATURE_SCHEMA)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--api-timeout-sec", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="Build reports without calling the API.")
    args = parser.parse_args()

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    schema = load_feature_schema(args.feature_schema)

    if args.source == "test_scaled_npz":
        X, y, row_ids = load_test_npz(args.test_npz)
    else:
        X, y, row_ids = load_deployment_15(args.deployment_15, schema)

    X, y, row_ids = select_rows(X, y, row_ids, args.max_rows, args.use_qga_mask, schema)

    prediction_rows: list[dict[str, Any]] = []
    valid_true: list[int] = []
    valid_pred: list[int] = []
    latencies: list[float] = []
    errors = 0

    for row_index, features in enumerate(X):
        true_label = int(y[row_index]) if y is not None else None
        if args.dry_run:
            response = None
            error = "dry_run_no_api_call"
            latency_ms = 0.0
        else:
            response, error, latency_ms = post_predict(
                args.api_url,
                [float(value) for value in features.tolist()],
                args.api_timeout_sec,
            )
            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

        prediction = response.get("prediction") if response else None
        probability_attack = response.get("probability_attack") if response else None
        label_pred = response.get("label") if response else None
        if error:
            errors += 1
        else:
            latencies.append(float(latency_ms))
            if true_label is not None and prediction is not None:
                valid_true.append(true_label)
                valid_pred.append(int(prediction))

        prediction_rows.append(
            {
                "row_index": row_index,
                "row_id": row_ids[row_index],
                "label_true": true_label,
                "prediction": prediction,
                "probability_attack": probability_attack,
                "label_pred": label_pred,
                "latency_ms": round(float(latency_ms), 6),
                "error": error,
            }
        )

    online_metrics = compute_binary_metrics(valid_true, valid_pred) if valid_true and valid_pred else {}
    summary = {
        "phase": "P15",
        "mode": "online_http_replay",
        "api_url": args.api_url,
        "source": args.source,
        "dry_run": bool(args.dry_run),
        "use_qga_mask": bool(args.use_qga_mask),
        "selected_mask_id": "conservative_seed_42" if args.use_qga_mask else None,
        "input_dim_sent": int(X.shape[1]),
        "rows_attempted": int(len(X)),
        "rows_predicted": int(len(valid_pred) if y is not None else len([r for r in prediction_rows if not r["error"]])),
        "labels_available": y is not None,
        "errors": int(errors),
        "latency": latency_summary(latencies),
        "online_metrics": online_metrics,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
        "test_used_for_online_deployment_replay_only": True,
        "accepted": bool(not args.dry_run and errors == 0 and len(prediction_rows) > 0),
    }

    write_predictions_csv(args.reports_dir / "p15_online_http_replay_predictions.csv", prediction_rows)
    (args.reports_dir / "p15_online_http_replay_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_table(args.reports_dir / "p15_online_http_replay_table.md", summary)

    print(json.dumps(summary, indent=2))
    return 0 if args.dry_run or errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
