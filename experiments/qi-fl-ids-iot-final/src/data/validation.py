"""Lightweight P1 validation for the final CIC-IoT parquet dataset."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from .label_mapping import (
    BENIGN_LABEL_ID,
    BENIGN_LABEL_NAME,
    EXPECTED_COLUMNS,
    EXPECTED_FEATURES,
    LABEL_COLUMN,
    build_id_to_label,
    build_label_to_binary,
    build_label_to_family,
    json_ready_id_to_label,
    load_label_mapping,
    validate_label_mapping,
)


@dataclass(frozen=True)
class ValidationRun:
    """In-memory result returned by the P1 validator."""

    summary: dict[str, Any]
    profile: dict[str, Any]
    errors: list[str]
    warnings: list[str]
    generated_files: list[str]

    @property
    def accepted(self) -> bool:
        return not self.errors


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the P1 YAML config."""
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return data


def _repo_path(repo_root: Path, relative_path: str) -> Path:
    return (repo_root / relative_path).resolve()


def _rel(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")


def _arrow_type_is_numeric(arrow_type: pa.DataType) -> bool:
    return (
        pa.types.is_integer(arrow_type)
        or pa.types.is_floating(arrow_type)
        or pa.types.is_decimal(arrow_type)
    )


def _safe_float(value: float | np.floating[Any] | None) -> float | None:
    if value is None:
        return None
    value_float = float(value)
    if math.isnan(value_float) or math.isinf(value_float):
        return None
    return value_float


def _array_to_float64(array: pa.Array) -> np.ndarray:
    values = array.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float64)


def _missing_count(array: pa.Array) -> int:
    missing = pc.is_null(array, nan_is_null=True)
    return int(pc.sum(missing).as_py() or 0)


def _figure_bar(
    labels: list[str],
    values: list[int | float],
    output_path: Path,
    title: str,
    xlabel: str,
    *,
    horizontal: bool = False,
    color: str = "#3B82F6",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if horizontal:
        height = max(5.0, 0.32 * len(labels))
        fig, ax = plt.subplots(figsize=(11, height))
        positions = np.arange(len(labels))
        ax.barh(positions, values, color=color)
        ax.set_yticks(positions, labels=labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(labels, values, color=color)
        ax.set_ylabel(xlabel)
        ax.tick_params(axis="x", rotation=20)
    ax.set_title(title)
    ax.grid(axis="x" if horizontal else "y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _feature_stats_init() -> dict[str, dict[str, Any]]:
    return {
        feature: {
            "finite_count": 0,
            "missing_count": 0,
            "infinite_count": 0,
            "min": None,
            "max": None,
            "sum": 0.0,
            "sum_sq": 0.0,
        }
        for feature in EXPECTED_FEATURES
    }


def _finalize_feature_stats(raw_stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for feature, stats in raw_stats.items():
        count = int(stats["finite_count"])
        if count:
            mean = stats["sum"] / count
            variance = max((stats["sum_sq"] / count) - (mean * mean), 0.0)
            std = math.sqrt(variance)
        else:
            mean = None
            std = None
        finalized[feature] = {
            "finite_count": count,
            "missing_count": int(stats["missing_count"]),
            "infinite_count": int(stats["infinite_count"]),
            "min": _safe_float(stats["min"]),
            "max": _safe_float(stats["max"]),
            "mean": _safe_float(mean),
            "std": _safe_float(std),
        }
    return finalized


def _profile_row_groups(
    parquet_file: pq.ParquetFile,
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    total_rows = int(parquet_file.metadata.num_rows)
    missing_counts = {column: 0 for column in EXPECTED_COLUMNS}
    infinite_counts = {column: 0 for column in EXPECTED_COLUMNS}
    label_counts: dict[int, int] = {}
    feature_stats = _feature_stats_init()

    for row_group_index in range(parquet_file.metadata.num_row_groups):
        table = parquet_file.read_row_group(row_group_index, columns=EXPECTED_COLUMNS)
        for column in EXPECTED_COLUMNS:
            array = table.column(column).combine_chunks()
            missing_counts[column] += _missing_count(array)

            arrow_type = array.type
            numeric = _array_to_float64(array)
            finite_mask = np.isfinite(numeric)
            inf_count = int(np.isinf(numeric).sum()) if pa.types.is_floating(arrow_type) else 0
            infinite_counts[column] += inf_count

            if column == LABEL_COLUMN:
                label_values = numeric[finite_mask].astype(np.int64, copy=False)
                unique_ids, counts = np.unique(label_values, return_counts=True)
                for label_id, count in zip(unique_ids.tolist(), counts.tolist(), strict=True):
                    label_counts[int(label_id)] = label_counts.get(int(label_id), 0) + int(count)
                continue

            finite_values = numeric[finite_mask]
            stats = feature_stats[column]
            stats["missing_count"] += missing_counts[column] - stats["missing_count"]
            stats["infinite_count"] += inf_count
            stats["finite_count"] += int(finite_values.size)
            if finite_values.size:
                min_value = float(np.min(finite_values))
                max_value = float(np.max(finite_values))
                stats["min"] = min_value if stats["min"] is None else min(stats["min"], min_value)
                stats["max"] = max_value if stats["max"] is None else max(stats["max"], max_value)
                stats["sum"] += float(np.sum(finite_values, dtype=np.float64))
                stats["sum_sq"] += float(np.sum(finite_values * finite_values, dtype=np.float64))

    class_distribution_by_id = {
        str(label_id): {
            "label_name": id_to_label.get(label_id, "__UNKNOWN__"),
            "count": int(label_counts.get(label_id, 0)),
            "ratio": (label_counts.get(label_id, 0) / total_rows) if total_rows else 0.0,
        }
        for label_id in sorted(id_to_label)
    }
    class_distribution_by_name = {
        payload["label_name"]: {
            "label_id": int(label_id),
            "count": payload["count"],
            "ratio": payload["ratio"],
        }
        for label_id, payload in class_distribution_by_id.items()
    }

    class_rows = [
        {
            "label_id": int(label_id),
            "label_name": payload["label_name"],
            "count": payload["count"],
            "ratio": payload["ratio"],
        }
        for label_id, payload in class_distribution_by_id.items()
    ]
    nonzero_counts = [row["count"] for row in class_rows if row["count"] > 0]
    imbalance_ratio = (
        max(nonzero_counts) / min(nonzero_counts) if nonzero_counts else None
    )

    return {
        "rows_profiled": total_rows,
        "label_counts": label_counts,
        "class_distribution_by_id": class_distribution_by_id,
        "class_distribution_by_name": class_distribution_by_name,
        "top_10_classes": sorted(class_rows, key=lambda row: (-row["count"], row["label_id"]))[:10],
        "bottom_10_classes": sorted(class_rows, key=lambda row: (row["count"], row["label_id"]))[:10],
        "imbalance_ratio": _safe_float(imbalance_ratio),
        "missing_count_by_column": {column: int(value) for column, value in missing_counts.items()},
        "missing_ratio_by_column": {
            column: (value / total_rows if total_rows else 0.0)
            for column, value in missing_counts.items()
        },
        "infinite_count_by_column": {
            column: int(value) for column, value in infinite_counts.items()
        },
        "infinite_ratio_by_column": {
            column: (value / total_rows if total_rows else 0.0)
            for column, value in infinite_counts.items()
        },
        "feature_stats": _finalize_feature_stats(feature_stats),
    }


def _build_binary_distribution(
    label_counts: dict[int, int],
    id_to_label: dict[int, str],
) -> dict[str, dict[str, Any]]:
    totals = {
        "normal": {"binary_label": 0, "count": 0, "ratio": 0.0},
        "attack": {"binary_label": 1, "count": 0, "ratio": 0.0},
    }
    total_rows = sum(label_counts.values())
    for label_id, count in label_counts.items():
        label_name = id_to_label.get(label_id)
        bucket = "normal" if label_name == BENIGN_LABEL_NAME else "attack"
        totals[bucket]["count"] += int(count)
    if total_rows:
        for payload in totals.values():
            payload["ratio"] = payload["count"] / total_rows
    return totals


def _build_family_distribution(
    label_counts: dict[int, int],
    id_to_label: dict[int, str],
    label_to_family: dict[str, str],
) -> dict[str, dict[str, Any]]:
    family_counts: dict[str, int] = {}
    total_rows = sum(label_counts.values())
    for label_id, count in label_counts.items():
        label_name = id_to_label.get(label_id)
        if label_name is None:
            continue
        family = label_to_family[label_name]
        family_counts[family] = family_counts.get(family, 0) + int(count)
    return {
        family: {
            "count": count,
            "ratio": count / total_rows if total_rows else 0.0,
        }
        for family, count in sorted(family_counts.items())
    }


def _generate_figures(
    figures_dir: Path,
    profile: dict[str, Any],
) -> list[Path]:
    binary = profile["distributions"]["binary"]
    family = profile["distributions"]["family"]
    class_by_id = profile["distributions"]["class_by_id"]
    missing = profile["quality"]["missing_count_by_column"]
    infinite = profile["quality"]["infinite_count_by_column"]
    dtypes = profile["schema"]["dtypes"]

    outputs = [
        figures_dir / "01_binary_distribution.png",
        figures_dir / "02_class_distribution_34.png",
        figures_dir / "03_family_distribution.png",
        figures_dir / "04_missing_values_by_column.png",
        figures_dir / "05_infinite_values_by_column.png",
        figures_dir / "06_feature_types.png",
    ]

    _figure_bar(
        list(binary.keys()),
        [payload["count"] for payload in binary.values()],
        outputs[0],
        "L1 binary distribution",
        "Rows",
        color="#2563EB",
    )

    class_labels = [
        f"{label_id} - {payload['label_name']}"
        for label_id, payload in class_by_id.items()
    ]
    class_values = [payload["count"] for payload in class_by_id.values()]
    _figure_bar(
        class_labels,
        class_values,
        outputs[1],
        "34-class label distribution",
        "Rows",
        horizontal=True,
        color="#0F766E",
    )

    _figure_bar(
        list(family.keys()),
        [payload["count"] for payload in family.values()],
        outputs[2],
        "L2 family distribution",
        "Rows",
        color="#7C3AED",
    )

    _figure_bar(
        list(missing.keys()),
        list(missing.values()),
        outputs[3],
        "Missing values by column",
        "Missing values",
        horizontal=True,
        color="#B45309",
    )

    _figure_bar(
        list(infinite.keys()),
        list(infinite.values()),
        outputs[4],
        "Infinite values by column",
        "Infinite values",
        horizontal=True,
        color="#BE123C",
    )

    dtype_counts: dict[str, int] = {}
    for dtype in dtypes.values():
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    _figure_bar(
        list(dtype_counts.keys()),
        list(dtype_counts.values()),
        outputs[5],
        "Feature and label column types",
        "Columns",
        color="#475569",
    )

    return outputs


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, sep, *body])


def _write_markdown_report(
    docs_path: Path,
    summary: dict[str, Any],
    profile: dict[str, Any],
    generated_files: list[str],
) -> None:
    accepted = bool(summary["accepted"])
    conclusion = "P1 est validée." if accepted else "P1 n'est pas validée."
    warnings = summary.get("warnings", [])
    errors = summary.get("errors", [])

    class_rows = [
        {
            "label_id": label_id,
            "label": payload["label_name"],
            "count": payload["count"],
            "ratio": f"{payload['ratio']:.6f}",
        }
        for label_id, payload in profile["distributions"]["class_by_id"].items()
    ]
    family_rows = [
        {"family": family, "count": payload["count"], "ratio": f"{payload['ratio']:.6f}"}
        for family, payload in profile["distributions"]["family"].items()
    ]
    binary_rows = [
        {
            "name": name,
            "binary_label": payload["binary_label"],
            "count": payload["count"],
            "ratio": f"{payload['ratio']:.6f}",
        }
        for name, payload in profile["distributions"]["binary"].items()
    ]
    stats_rows = [
        {
            "feature": feature,
            "min": stats["min"],
            "max": stats["max"],
            "mean": stats["mean"],
            "std": stats["std"],
        }
        for feature, stats in profile["feature_statistics"].items()
    ]

    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(
        "\n".join(
            [
                "# P1 — Data Validation Report",
                "",
                "## 1. Objectif",
                "",
                "Valider le dataset final CIC-IoT avant preprocessing, split, scaling ou entraînement.",
                "",
                "## 2. Dataset utilisé",
                "",
                f"- Parquet prioritaire : `{summary['dataset']['parquet_path']}`",
                f"- CSV secondaire : `{summary['dataset']['csv_path']}`",
                f"- Mapping labels : `{summary['dataset']['label_mapping_path']}`",
                "",
                "## 3. Méthode de validation",
                "",
                "La validation utilise les métadonnées Parquet puis une lecture par row groups et colonnes. Le CSV complet n'est pas chargé.",
                "",
                "## 4. Schéma du dataset",
                "",
                f"- Shape : `{summary['dataset']['rows']} x {summary['dataset']['columns']}`",
                f"- Row groups : `{summary['dataset']['row_groups']}`",
                f"- Colonne label : `{summary['schema']['label_column']}`",
                "",
                "## 5. Features validées",
                "",
                f"- Nombre de features : `{summary['schema']['num_features']}`",
                f"- Features numériques : `{summary['criteria']['numeric_features']}`",
                f"- `label_id` exclu des features : `{summary['criteria']['label_excluded_from_features']}`",
                "",
                "## 6. Mapping des labels",
                "",
                f"- Classes : `{summary['schema']['num_classes']}`",
                f"- `{BENIGN_LABEL_NAME}` = `{BENIGN_LABEL_ID}` : `{summary['criteria']['benign_label_id']}`",
                f"- Label ids inconnus : `{summary['label_mapping']['unknown_label_ids']}`",
                "",
                "## 7. Mapping binaire L1",
                "",
                "Pour L1, `normal = 0` et `attack = 1`. Le `label_id` original `1` (`BenignTraffic`) devient `binary_label = 0`; toutes les autres classes deviennent `binary_label = 1`.",
                "",
                _markdown_table(binary_rows, ["name", "binary_label", "count", "ratio"]),
                "",
                "## 8. Mapping familles L2",
                "",
                _markdown_table(family_rows, ["family", "count", "ratio"]),
                "",
                "## 9. Distribution 34 classes L3",
                "",
                _markdown_table(class_rows, ["label_id", "label", "count", "ratio"]),
                "",
                "## 10. Valeurs manquantes",
                "",
                f"- Total NaN/null : `{summary['quality']['total_missing_values']}`",
                "",
                "## 11. Valeurs infinies",
                "",
                f"- Total ±inf : `{summary['quality']['total_infinite_values']}`",
                "",
                "## 12. Statistiques des features",
                "",
                _markdown_table(stats_rows, ["feature", "min", "max", "mean", "std"]),
                "",
                "## 13. Artefacts générés",
                "",
                "\n".join(f"- `{path}`" for path in generated_files if "/artifacts/" in path),
                "",
                "## 14. Figures générées",
                "",
                "\n".join(f"- `{path}`" for path in generated_files if path.endswith(".png")),
                "",
                "## 15. Risques restants",
                "",
                "\n".join(f"- WARNING: {warning}" for warning in warnings) if warnings else "- Aucun warning bloquant.",
                "",
                "## 16. Critères d’acceptation",
                "",
                _markdown_table(
                    [
                        {"critere": key, "ok": value}
                        for key, value in summary["criteria"].items()
                    ],
                    ["critere", "ok"],
                ),
                "",
                "## 17. Conclusion P1",
                "",
                conclusion,
                "",
                "\n".join(f"- ERROR: {error}" for error in errors),
                "",
            ]
        ),
        encoding="utf-8",
    )


def validate(config_path: Path) -> ValidationRun:
    """Run P1 validation and write all lightweight reports/artifacts."""
    config = load_config(config_path)
    repo_root = _repo_path(Path.cwd(), config.get("project_root", "."))

    dataset_cfg = config["dataset"]
    schema_cfg = config["schema"]
    outputs_cfg = config["outputs"]

    parquet_path = _repo_path(repo_root, dataset_cfg["parquet_path"])
    csv_path = _repo_path(repo_root, dataset_cfg["csv_path"])
    label_mapping_path = _repo_path(repo_root, dataset_cfg["label_mapping_path"])
    reports_dir = _repo_path(repo_root, outputs_cfg["reports_dir"])
    figures_dir = _repo_path(repo_root, outputs_cfg["figures_dir"])
    artifacts_dir = _repo_path(repo_root, outputs_cfg["artifacts_dir"])
    docs_dir = _repo_path(repo_root, outputs_cfg["docs_dir"])

    for path in [
        reports_dir,
        figures_dir,
        artifacts_dir / "features",
        artifacts_dir / "mappings",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    warnings: list[str] = []
    generated: list[Path] = []

    if not parquet_path.exists():
        errors.append(f"parquet file not found: {dataset_cfg['parquet_path']}")
    if not label_mapping_path.exists():
        errors.append(f"label mapping not found: {dataset_cfg['label_mapping_path']}")
    if not csv_path.exists():
        warnings.append(f"csv file not found: {dataset_cfg['csv_path']}")
    if errors:
        summary = {"accepted": False, "errors": errors, "warnings": warnings}
        profile = {"errors": errors, "warnings": warnings}
        return ValidationRun(summary, profile, errors, warnings, [])

    parquet_file = pq.ParquetFile(parquet_path)
    metadata = parquet_file.metadata
    schema = parquet_file.schema_arrow
    columns = list(schema.names)
    dtypes = {field.name: str(field.type) for field in schema}
    label_mapping = load_label_mapping(label_mapping_path)
    id_to_label = build_id_to_label(label_mapping)
    label_to_family = build_label_to_family(label_mapping)
    label_to_binary = build_label_to_binary(label_mapping)

    expected_rows = int(schema_cfg["expected_rows"])
    expected_columns = int(schema_cfg["expected_columns"])
    label_column = str(schema_cfg["label_column"])
    features = [column for column in columns if column != label_column]

    if metadata.num_rows != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {metadata.num_rows}")
    if metadata.num_columns != expected_columns:
        errors.append(f"expected {expected_columns} columns, found {metadata.num_columns}")
    if label_column not in columns:
        errors.append(f"label column missing: {label_column}")
    if features != EXPECTED_FEATURES:
        errors.append("feature order or names do not match the expected 28-feature schema")
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in columns]
    unexpected_columns = [column for column in columns if column not in EXPECTED_COLUMNS]
    if missing_columns:
        errors.append(f"missing columns: {missing_columns}")
    if unexpected_columns:
        errors.append(f"unexpected columns: {unexpected_columns}")
    if len(features) != int(schema_cfg["expected_num_features"]):
        errors.append(f"expected 28 features, found {len(features)}")

    feature_types = {feature: schema.field(feature).type for feature in features if feature in columns}
    non_numeric_features = [
        feature for feature, arrow_type in feature_types.items() if not _arrow_type_is_numeric(arrow_type)
    ]
    if non_numeric_features:
        errors.append(f"non-numeric features: {non_numeric_features}")

    errors.extend(validate_label_mapping(label_mapping))

    row_group_profile = _profile_row_groups(parquet_file, id_to_label)
    label_counts = row_group_profile["label_counts"]
    observed_ids = sorted(label_counts)
    mapping_ids = set(id_to_label)
    unknown_label_ids = [label_id for label_id in observed_ids if label_id not in mapping_ids]
    if unknown_label_ids:
        errors.append(f"dataset contains unknown label ids: {unknown_label_ids}")
    absent_mapping_ids = [label_id for label_id in sorted(mapping_ids) if label_id not in label_counts]
    if absent_mapping_ids:
        warnings.append(f"mapping ids absent from dataset distribution: {absent_mapping_ids}")

    binary_distribution = _build_binary_distribution(label_counts, id_to_label)
    family_distribution = _build_family_distribution(label_counts, id_to_label, label_to_family)
    total_missing = int(sum(row_group_profile["missing_count_by_column"].values()))
    total_infinite = int(sum(row_group_profile["infinite_count_by_column"].values()))

    profile: dict[str, Any] = {
        "dataset": {
            "source_priority": dataset_cfg["source_priority"],
            "parquet_path": dataset_cfg["parquet_path"],
            "csv_path": dataset_cfg["csv_path"],
            "label_mapping_path": dataset_cfg["label_mapping_path"],
            "parquet_size_bytes": parquet_path.stat().st_size,
            "csv_size_bytes": csv_path.stat().st_size if csv_path.exists() else None,
            "rows": metadata.num_rows,
            "columns": metadata.num_columns,
            "row_groups": metadata.num_row_groups,
        },
        "schema": {
            "columns": columns,
            "features": features,
            "label_column": label_column,
            "dtypes": dtypes,
            "non_numeric_features": non_numeric_features,
            "missing_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
        },
        "label_mapping": {
            "label_to_id": label_mapping,
            "id_to_label": json_ready_id_to_label(id_to_label),
            "label_to_binary": label_to_binary,
            "label_to_family": label_to_family,
            "observed_label_ids": observed_ids,
            "unknown_label_ids": unknown_label_ids,
            "mapping_ids_absent_from_dataset": absent_mapping_ids,
        },
        "distributions": {
            "class_by_id": row_group_profile["class_distribution_by_id"],
            "class_by_name": row_group_profile["class_distribution_by_name"],
            "binary": binary_distribution,
            "family": family_distribution,
            "top_10_classes": row_group_profile["top_10_classes"],
            "bottom_10_classes": row_group_profile["bottom_10_classes"],
            "imbalance_ratio": row_group_profile["imbalance_ratio"],
        },
        "quality": {
            "missing_count_by_column": row_group_profile["missing_count_by_column"],
            "missing_ratio_by_column": row_group_profile["missing_ratio_by_column"],
            "infinite_count_by_column": row_group_profile["infinite_count_by_column"],
            "infinite_ratio_by_column": row_group_profile["infinite_ratio_by_column"],
            "total_missing_values": total_missing,
            "total_infinite_values": total_infinite,
        },
        "feature_statistics": row_group_profile["feature_stats"],
    }

    criteria = {
        "parquet_found": parquet_path.exists(),
        "label_mapping_found": label_mapping_path.exists(),
        "shape_matches": metadata.num_rows == expected_rows and metadata.num_columns == expected_columns,
        "features_count_28": len(features) == int(schema_cfg["expected_num_features"]),
        "label_id_present": label_column in columns,
        "numeric_features": not non_numeric_features,
        "classes_count_34": len(label_mapping) == int(schema_cfg["expected_num_classes"]),
        "benign_label_id": label_mapping.get(BENIGN_LABEL_NAME) == BENIGN_LABEL_ID,
        "dataset_label_ids_known": not unknown_label_ids,
        "missing_values_computed": len(row_group_profile["missing_count_by_column"]) == len(columns),
        "infinite_values_computed": len(row_group_profile["infinite_count_by_column"]) == len(columns),
        "binary_distribution_generated": bool(binary_distribution),
        "family_distribution_generated": bool(family_distribution),
        "class_distribution_generated": len(row_group_profile["class_distribution_by_id"]) == len(label_mapping),
        "label_excluded_from_features": label_column not in features,
    }

    summary: dict[str, Any] = {
        "accepted": not errors and all(criteria.values()),
        "errors": errors,
        "warnings": warnings,
        "dataset": profile["dataset"],
        "schema": {
            "num_features": len(features),
            "num_classes": len(label_mapping),
            "num_families": len(family_distribution),
            "label_column": label_column,
        },
        "label_mapping": {
            "benign_label_name": BENIGN_LABEL_NAME,
            "benign_label_id": label_mapping.get(BENIGN_LABEL_NAME),
            "unknown_label_ids": unknown_label_ids,
        },
        "quality": {
            "total_missing_values": total_missing,
            "total_infinite_values": total_infinite,
        },
        "distributions": {
            "binary": binary_distribution,
            "family": family_distribution,
            "top_10_classes": row_group_profile["top_10_classes"],
            "bottom_10_classes": row_group_profile["bottom_10_classes"],
            "imbalance_ratio": row_group_profile["imbalance_ratio"],
        },
        "criteria": criteria,
    }
    summary["accepted"] = not errors and all(criteria.values())

    feature_names_path = artifacts_dir / "features" / "feature_names.json"
    mapping_path = artifacts_dir / "mappings" / "label_mapping.json"
    id_to_label_path = artifacts_dir / "mappings" / "id_to_label.json"
    family_path = artifacts_dir / "mappings" / "label_to_family.json"
    binary_path = artifacts_dir / "mappings" / "label_to_binary.json"
    summary_path = reports_dir / "data_validation_summary.json"
    profile_path = reports_dir / "data_validation_profile.json"
    docs_path = docs_dir / "01_data_validation.md"

    for path, payload in [
        (feature_names_path, features),
        (mapping_path, label_mapping),
        (id_to_label_path, json_ready_id_to_label(id_to_label)),
        (family_path, label_to_family),
        (binary_path, label_to_binary),
        (summary_path, summary),
        (profile_path, profile),
    ]:
        _write_json(path, payload)
        generated.append(path)

    figures = _generate_figures(figures_dir, profile)
    generated.extend(figures)

    generated_rel = [_rel(path, repo_root) for path in generated]
    _write_markdown_report(docs_path, summary, profile, generated_rel)
    generated.append(docs_path)

    return ValidationRun(
        summary=summary,
        profile=profile,
        errors=errors,
        warnings=warnings,
        generated_files=[_rel(path, repo_root) for path in generated],
    )
