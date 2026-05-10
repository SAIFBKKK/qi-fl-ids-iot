"""P2 preprocessing for final CIC-IoT L1 and L2 datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .scaling import (
    RobustScalerFit,
    ScalingStatus,
    check_scaling_status,
    fit_robust_scaler_from_parquet,
    load_p1_feature_statistics,
    save_scaler,
    transform_parquet_to_npz,
)
from .splitting import (
    SPLIT_CODES,
    anti_leakage_report,
    assign_to_global,
    split_counts_by_key,
    stratified_train_val_test_split,
)

FAMILY_ID_MAP: dict[str, int] = {
    "BruteForce": 0,
    "DDoS": 1,
    "DoS": 2,
    "Malware": 3,
    "Mirai": 4,
    "Recon": 5,
    "Spoofing": 6,
    "Web-based": 7,
}


@dataclass(frozen=True)
class PreprocessingRun:
    """Result returned by the P2 preprocessing pipeline."""

    summary: dict[str, Any]
    profile: dict[str, Any]
    errors: list[str]
    warnings: list[str]
    generated_files: list[str]

    @property
    def accepted(self) -> bool:
        return not self.errors and bool(self.summary.get("accepted", False))


def load_config(config_path: Path) -> dict[str, Any]:
    """Load preprocessing YAML config."""

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return config


def _repo_path(repo_root: Path, relative_path: str) -> Path:
    return (repo_root / relative_path).resolve()


def _rel(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")


def _file_info(path: Path, repo_root: Path) -> dict[str, Any]:
    return {"path": _rel(path, repo_root), "size_bytes": int(path.stat().st_size)}


def _created_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_output_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _verify_p1(
    repo_root: Path,
    config: dict[str, Any],
    feature_names: list[str] | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    inputs = config["inputs"]
    schema = config["schema"]
    final_dir = _repo_path(repo_root, config["final_experiment_dir"])

    required_paths = {
        "p1_report": _repo_path(repo_root, inputs["p1_report"]),
        "feature_names": _repo_path(repo_root, inputs["feature_names_path"]),
        "label_mapping": _repo_path(repo_root, inputs["label_mapping_path"]),
        "id_to_label": _repo_path(repo_root, inputs["id_to_label_path"]),
        "label_to_binary": _repo_path(repo_root, inputs["label_to_binary_path"]),
        "label_to_family": _repo_path(repo_root, inputs["label_to_family_path"]),
        "parquet": _repo_path(repo_root, inputs["parquet_path"]),
    }

    for name, path in required_paths.items():
        if not path.exists():
            errors.append(f"missing required P1/input artifact: {name} at {_rel(path, repo_root)}")

    p1_summary_path = final_dir / "outputs" / "reports" / "data_validation_summary.json"
    p1_summary: dict[str, Any] = {}
    if p1_summary_path.exists():
        p1_summary = _load_json(p1_summary_path)
        if not p1_summary.get("accepted", False):
            errors.append("P1 summary exists but is not accepted")
    else:
        warnings.append("P1 summary JSON not found; relying on docs/01_data_validation.md")

    p1_report_path = required_paths["p1_report"]
    if p1_report_path.exists():
        report_text = p1_report_path.read_text(encoding="utf-8")
        if "P1 est validée" not in report_text and "P1 est validee" not in report_text:
            warnings.append("P1 markdown report does not contain the expected validation sentence")

    expected_num_features = int(schema["expected_num_features"])
    if feature_names is not None and len(feature_names) != expected_num_features:
        errors.append(
            f"expected {expected_num_features} P1 features, found {len(feature_names)}"
        )

    return {
        "required_paths": {
            name: _rel(path, repo_root) for name, path in required_paths.items()
        },
        "p1_summary_path": _rel(p1_summary_path, repo_root)
        if p1_summary_path.exists()
        else None,
        "p1_accepted": bool(p1_summary.get("accepted", True)),
        "feature_count": len(feature_names or []),
    }, errors, warnings


def _read_label_ids(parquet_path: Path, label_column: str) -> tuple[np.ndarray, dict[str, int]]:
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    labels = np.empty(total_rows, dtype=np.int16)

    cursor = 0
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        table = parquet_file.read_row_group(row_group_index, columns=[label_column])
        values = table.column(label_column).combine_chunks().to_numpy(zero_copy_only=False)
        values = np.asarray(values, dtype=np.int16)
        labels[cursor : cursor + values.size] = values
        cursor += values.size

    metadata = {
        "num_rows": int(parquet_file.metadata.num_rows),
        "num_columns": int(parquet_file.metadata.num_columns),
        "num_row_groups": int(parquet_file.metadata.num_row_groups),
    }
    return labels, metadata


def _label_counts(label_ids: np.ndarray, id_to_label: dict[int, str]) -> dict[str, dict[str, Any]]:
    values, counts = np.unique(label_ids, return_counts=True)
    return {
        str(int(label_id)): {
            "label_name": id_to_label[int(label_id)],
            "count": int(count),
        }
        for label_id, count in zip(values, counts)
    }


def _build_lookup_arrays(
    id_to_label: dict[int, str],
    label_to_binary: dict[str, dict[str, Any]],
    label_to_family: dict[str, str],
) -> dict[str, np.ndarray]:
    max_label_id = max(id_to_label)
    label_name_lookup = np.empty(max_label_id + 1, dtype=object)
    binary_label_lookup = np.full(max_label_id + 1, -1, dtype=np.int8)
    binary_name_lookup = np.empty(max_label_id + 1, dtype=object)
    family_name_lookup = np.empty(max_label_id + 1, dtype=object)
    family_id_lookup = np.full(max_label_id + 1, -1, dtype=np.int8)

    for label_id, label_name in sorted(id_to_label.items()):
        family_name = label_to_family[label_name]
        label_name_lookup[label_id] = label_name
        binary_label_lookup[label_id] = int(label_to_binary[label_name]["binary_label"])
        binary_name_lookup[label_id] = str(label_to_binary[label_name]["binary_name"])
        family_name_lookup[label_id] = family_name
        family_id_lookup[label_id] = FAMILY_ID_MAP.get(family_name, -1)

    return {
        "label_name": label_name_lookup,
        "binary_label": binary_label_lookup,
        "binary_name": binary_name_lookup,
        "family_name": family_name_lookup,
        "family_id": family_id_lookup,
    }


def _build_l1_sample(
    label_ids: np.ndarray,
    *,
    benign_label_id: int,
    attack_samples_per_class: int,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    errors: list[str] = []
    rng = np.random.default_rng(random_seed)
    selected_parts: list[np.ndarray] = []
    normal_rows = np.flatnonzero(label_ids == benign_label_id)
    selected_parts.append(normal_rows)

    attack_class_ids = [
        int(label_id)
        for label_id in sorted(np.unique(label_ids).tolist())
        if int(label_id) != benign_label_id
    ]
    selected_attack_ids: list[int] = []

    for label_id in attack_class_ids:
        rows = np.flatnonzero(label_ids == label_id)
        if rows.size < attack_samples_per_class:
            errors.append(
                f"label_id {label_id} has only {rows.size} rows, cannot sample "
                f"{attack_samples_per_class}"
            )
            continue
        selected = rng.choice(rows, size=attack_samples_per_class, replace=False)
        selected_parts.append(selected)
        selected_attack_ids.append(label_id)

    selected_rows = np.sort(np.concatenate(selected_parts).astype(np.int64))
    selected_labels = label_ids[selected_rows]
    _, after_counts = np.unique(selected_labels, return_counts=True)

    report = {
        "random_seed": int(random_seed),
        "sampling_rules": {
            "normal": "keep all BenignTraffic rows",
            "attacks": f"sample exactly {attack_samples_per_class} rows per attack label_id",
            "replace": False,
        },
        "normal_count": int(normal_rows.size),
        "attack_count": int(np.sum(selected_labels != benign_label_id)),
        "total_count": int(selected_rows.size),
        "selected_attack_label_ids": selected_attack_ids,
        "after_count_values": [int(value) for value in after_counts.tolist()],
    }
    return selected_rows, report, errors


def _enrich_table(
    table: pa.Table,
    row_ids: np.ndarray,
    feature_names: list[str],
    label_column: str,
    lookups: dict[str, np.ndarray],
    *,
    include_binary: bool,
) -> pa.Table:
    label_values = np.asarray(
        table.column(label_column).combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int16,
    )
    label_names = lookups["label_name"][label_values]
    family_names = lookups["family_name"][label_values]
    family_ids = lookups["family_id"][label_values]

    arrays: list[pa.Array | pa.ChunkedArray] = [pa.array(row_ids, type=pa.int64())]
    names = ["row_id"]

    for feature in feature_names:
        arrays.append(table.column(feature))
        names.append(feature)

    arrays.extend(
        [
            table.column(label_column),
            pa.array(label_names, type=pa.string()),
        ]
    )
    names.extend([label_column, "label_name"])

    if include_binary:
        binary_labels = lookups["binary_label"][label_values]
        binary_names = lookups["binary_name"][label_values]
        arrays.extend(
            [
                pa.array(binary_labels, type=pa.int8()),
                pa.array(binary_names, type=pa.string()),
            ]
        )
        names.extend(["binary_label", "binary_name"])

    arrays.extend(
        [
            pa.array(family_names, type=pa.string()),
            pa.array(family_ids, type=pa.int8()),
        ]
    )
    names.extend(["family_name", "family_id"])
    return pa.Table.from_arrays(arrays, names=names)


def _write_preprocessed_splits(
    source_path: Path,
    output_dir: Path,
    feature_names: list[str],
    label_column: str,
    split_assignments: np.ndarray,
    lookups: dict[str, np.ndarray],
    *,
    include_binary: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        split_name: output_dir / f"{split_name}.parquet"
        for split_name in SPLIT_CODES
    }
    writers: dict[str, pq.ParquetWriter | None] = {name: None for name in SPLIT_CODES}

    parquet_file = pq.ParquetFile(source_path)
    columns = [*feature_names, label_column]
    offset = 0
    try:
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(row_group_index, columns=columns)
            n_rows = table.num_rows
            group_assignments = split_assignments[offset : offset + n_rows]

            for split_name, split_code in SPLIT_CODES.items():
                positions = np.flatnonzero(group_assignments == split_code)
                if positions.size == 0:
                    continue
                selected = table.take(pa.array(positions.astype(np.int64)))
                selected_row_ids = (offset + positions).astype(np.int64)
                enriched = _enrich_table(
                    selected,
                    selected_row_ids,
                    feature_names,
                    label_column,
                    lookups,
                    include_binary=include_binary,
                )
                if writers[split_name] is None:
                    writers[split_name] = pq.ParquetWriter(
                        paths[split_name],
                        enriched.schema,
                        compression="snappy",
                    )
                writers[split_name].write_table(enriched)
            offset += n_rows
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()

    return paths


def _split_file_infos(paths: dict[str, Path], repo_root: Path) -> dict[str, dict[str, Any]]:
    return {name: _file_info(path, repo_root) for name, path in paths.items()}


def _count_by_binary(label_ids: np.ndarray, benign_label_id: int) -> dict[str, dict[str, Any]]:
    normal = int(np.sum(label_ids == benign_label_id))
    attack = int(label_ids.size - normal)
    total = int(label_ids.size)
    return {
        "normal": {
            "binary_label": 0,
            "count": normal,
            "ratio": normal / total if total else 0.0,
        },
        "attack": {
            "binary_label": 1,
            "count": attack,
            "ratio": attack / total if total else 0.0,
        },
    }


def _family_distribution(
    label_ids: np.ndarray,
    family_id_lookup: np.ndarray,
    family_labels: dict[int, str],
) -> dict[str, dict[str, Any]]:
    family_ids = family_id_lookup[label_ids]
    valid = family_ids >= 0
    values, counts = np.unique(family_ids[valid], return_counts=True)
    total = int(np.sum(counts))
    return {
        family_labels[int(family_id)]: {
            "family_id": int(family_id),
            "count": int(count),
            "ratio": int(count) / total if total else 0.0,
        }
        for family_id, count in zip(values, counts)
    }


def _generate_bar(
    labels: list[str],
    values: list[int | float],
    path: Path,
    title: str,
    ylabel: str,
    *,
    horizontal: bool = False,
    color: str = "#2563EB",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if horizontal:
        fig, ax = plt.subplots(figsize=(12, max(5, len(labels) * 0.32)))
        positions = np.arange(len(labels))
        ax.barh(positions, values, color=color)
        ax.set_yticks(positions, labels=labels)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        positions = np.arange(len(labels))
        ax.bar(positions, values, color=color)
        ax.set_xticks(positions, labels=labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x" if horizontal else "y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _generate_grouped_split_figure(
    split_distribution: dict[str, dict[str, int]],
    path: Path,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(split_distribution.keys())
    splits = ["train", "val", "test"]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), 5.8))
    colors = ["#2563EB", "#16A34A", "#F97316"]
    for idx, split in enumerate(splits):
        values = [split_distribution[label][split] for label in labels]
        ax.bar(x + (idx - 1) * width, values, width, label=split, color=colors[idx])
    ax.set_xticks(x, labels=labels, rotation=25, ha="right")
    ax.set_ylabel("Rows")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _generate_binary_before_after_figure(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["normal", "attack"]
    x = np.arange(len(labels))
    width = 0.34
    before_values = [before[label]["count"] for label in labels]
    after_values = [after[label]["count"] for label in labels]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, before_values, width, label="before", color="#64748B")
    ax.bar(x + width / 2, after_values, width, label="after L1 sampling", color="#2563EB")
    ax.set_xticks(x, labels=labels)
    ax.set_ylabel("Rows")
    ax.set_title("L1 binary distribution before/after sampling")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _generate_scaling_status_figure(status: ScalingStatus, path: Path) -> None:
    indicators = status.indicators
    labels = ["raw indicators", "centered/bounded"]
    values = [
        int(indicators.get("raw_indicator_count", 0)),
        int(indicators.get("centered_bounded_features", 0)),
    ]
    _generate_bar(
        labels,
        values,
        path,
        f"Scaling status: {status.status}",
        "Feature count",
        color="#7C3AED" if status.status == "possibly_scaled" else "#DC2626",
    )


def _generate_scaling_before_after_figure(
    scaler_fit: RobustScalerFit | None,
    path: Path,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if scaler_fit is None:
        _generate_bar(["scaling skipped"], [0], path, title, "value", color="#64748B")
        return

    features = list(scaler_fit.feature_stats.keys())
    raw_iqr = [scaler_fit.feature_stats[feature]["iqr"] for feature in features]
    scaled_iqr = [1.0 for _ in features]

    x = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 0.2, raw_iqr, 0.4, label="raw train IQR", color="#DC2626")
    ax.bar(x + 0.2, scaled_iqr, 0.4, label="after RobustScaler IQR target", color="#16A34A")
    ax.set_yscale("symlog")
    ax.set_xticks(x, labels=features, rotation=70, ha="right")
    ax.set_ylabel("IQR (symlog)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _generate_pipeline_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [
        "P1 artifacts",
        "Parquet row_id",
        "L1 sampling",
        "L1 split + scaler",
        "L2 attack-only",
        "L2 split + scaler",
        "P2 manifests",
    ]
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")
    x_positions = np.linspace(0.05, 0.95, len(steps))
    for idx, (x_pos, step) in enumerate(zip(x_positions, steps)):
        ax.text(
            x_pos,
            0.55,
            step,
            ha="center",
            va="center",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#EFF6FF",
                "edgecolor": "#2563EB",
            },
        )
        if idx < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.045, 0.55),
                xytext=(x_pos + 0.045, 0.55),
                arrowprops={"arrowstyle": "->", "color": "#334155", "lw": 1.4},
            )
    ax.set_title("P2 preprocessing pipeline L1/L2", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_markdown_report(
    path: Path,
    summary: dict[str, Any],
    profile: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    accepted_text = "P2 est validée." if summary.get("accepted") else "P2 n'est pas validée."
    lines = [
        "# P2 — Preprocessing Report",
        "",
        "## 1. Objectif",
        "Préparer les datasets L1 binaire et L2 famille sans entraînement, sans Dirichlet et sans preprocessing hors périmètre.",
        "",
        "## 2. Entrées utilisées",
        f"- Source Parquet : `{summary['inputs']['parquet_path']}`",
        f"- Artefacts P1 : `{summary['inputs']['p1_artifacts_dir']}`",
        "",
        "## 3. Vérification P1",
        f"- P1 détectée : `{summary['criteria']['p1_validated_detected']}`",
        f"- Features confirmées : `{summary['schema']['feature_count']}`",
        "",
        "## 4. Vérification du statut de scaling",
        f"- Statut : `{summary['scaling']['status']}`",
        f"- Justification : {summary['scaling']['justification']}",
        f"- Scaling appliqué : `{summary['scaling']['applied']}`",
        "",
        "## 5. Construction du dataset L1 binaire équilibré",
        f"- Total : `{summary['l1_binary']['total_count']}`",
        f"- Normal : `{summary['l1_binary']['normal_count']}`",
        f"- Attack : `{summary['l1_binary']['attack_count']}`",
        f"- Sampling attaques : `{summary['l1_binary']['attack_samples_per_class']}` par classe.",
        "",
        "## 6. Split L1 train/val/test",
        f"- Counts : `{summary['l1_binary']['split_counts']}`",
        "",
        "## 7. Scaling L1 train-only",
        f"- Scaler : `{summary['l1_binary']['scaler_path']}`",
        f"- NPZ : `{summary['l1_binary']['npz_files']}`",
        "",
        "## 8. Construction du dataset L2 family attack-only",
        f"- Total : `{summary['l2_family']['total_count']}`",
        f"- Familles : `{summary['l2_family']['families']}`",
        f"- Sampling : `{summary['l2_family']['sampling']}`",
        "",
        "## 9. Split L2 train/val/test",
        f"- Counts : `{summary['l2_family']['split_counts']}`",
        "",
        "## 10. Scaling L2 train-only",
        f"- Scaler : `{summary['l2_family']['scaler_path']}`",
        f"- NPZ : `{summary['l2_family']['npz_files']}`",
        "",
        "## 11. Anti-leakage",
        f"- L1 : `{summary['l1_binary']['anti_leakage_result']}`",
        f"- L2 : `{summary['l2_family']['anti_leakage_result']}`",
        "",
        "## 12. Artefacts générés",
    ]
    for artifact in summary["generated_artifacts"]:
        lines.append(f"- `{artifact}`")
    lines.extend(["", "## 13. Figures générées"])
    for figure in summary["figures"]:
        lines.append(f"- `{figure}`")
    lines.extend(
        [
            "",
            "## 14. Risques restants",
        ]
    )
    if summary["warnings"]:
        lines.extend([f"- {warning}" for warning in summary["warnings"]])
    else:
        lines.append("- Aucun warning restant.")
    lines.extend(["", "## 15. Critères d’acceptation", "", "| critere | ok |", "| --- | --- |"])
    for key, value in summary["criteria"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## 16. Conclusion P2", "", accepted_text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _initial_failure_run(
    repo_root: Path,
    reports_dir: Path,
    errors: list[str],
    warnings: list[str],
) -> PreprocessingRun:
    summary = {
        "accepted": False,
        "errors": errors,
        "warnings": warnings,
        "criteria": {},
    }
    profile = {"errors": errors, "warnings": warnings}
    _write_json(reports_dir / "preprocessing_summary.json", summary)
    _write_json(reports_dir / "preprocessing_profile.json", profile)
    return PreprocessingRun(
        summary=summary,
        profile=profile,
        errors=errors,
        warnings=warnings,
        generated_files=[
            _rel(reports_dir / "preprocessing_summary.json", repo_root),
            _rel(reports_dir / "preprocessing_profile.json", repo_root),
        ],
    )


def run_preprocessing(config_path: Path) -> PreprocessingRun:
    """Run P2 preprocessing and materialize all required lightweight reports."""

    repo_root = Path.cwd().resolve()
    config = load_config(config_path)
    inputs = config["inputs"]
    schema = config["schema"]
    split_config = config["split"]
    l1_config = config["l1_binary"]
    l2_config = config["l2_family"]
    scaling_config = config["scaling"]
    output_config = config["outputs"]

    final_dir = _repo_path(repo_root, config["final_experiment_dir"])
    parquet_path = _repo_path(repo_root, inputs["parquet_path"])
    feature_names_path = _repo_path(repo_root, inputs["feature_names_path"])
    reports_dir = _repo_path(repo_root, output_config["reports_dir"])
    figures_dir = _repo_path(repo_root, output_config["figures_dir"])
    preprocessed_dir = _repo_path(repo_root, output_config["preprocessed_dir"])
    scalers_dir = _repo_path(repo_root, output_config["scalers_dir"])
    artifacts_preprocessing_dir = final_dir / "outputs" / "artifacts" / "preprocessing"
    l1_dir = preprocessed_dir / "l1_binary"
    l2_dir = preprocessed_dir / "l2_family"

    _ensure_output_dirs(
        [
            reports_dir,
            figures_dir,
            preprocessed_dir,
            l1_dir,
            l2_dir,
            scalers_dir,
            artifacts_preprocessing_dir,
        ]
    )

    errors: list[str] = []
    warnings: list[str] = []
    generated_files: list[str] = []

    feature_names: list[str] | None = None
    if feature_names_path.exists():
        feature_names = _load_json(feature_names_path)
        if not isinstance(feature_names, list) or not all(
            isinstance(item, str) for item in feature_names
        ):
            errors.append("feature_names.json must contain a list of strings")
            feature_names = None

    p1_check, p1_errors, p1_warnings = _verify_p1(repo_root, config, feature_names)
    errors.extend(p1_errors)
    warnings.extend(p1_warnings)
    if errors:
        return _initial_failure_run(repo_root, reports_dir, errors, warnings)

    assert feature_names is not None
    label_mapping = _load_json(_repo_path(repo_root, inputs["label_mapping_path"]))
    raw_id_to_label = _load_json(_repo_path(repo_root, inputs["id_to_label_path"]))
    id_to_label = {int(key): value for key, value in raw_id_to_label.items()}
    label_to_binary = _load_json(_repo_path(repo_root, inputs["label_to_binary_path"]))
    label_to_family = _load_json(_repo_path(repo_root, inputs["label_to_family_path"]))
    lookups = _build_lookup_arrays(id_to_label, label_to_binary, label_to_family)

    label_column = str(schema["label_column"])
    benign_label_id = int(schema["benign_label_id"])
    random_seed = int(split_config["random_seed"])
    train_ratio = float(split_config["train_ratio"])
    val_ratio = float(split_config["val_ratio"])
    test_ratio = float(split_config["test_ratio"])

    label_ids, parquet_metadata = _read_label_ids(parquet_path, label_column)
    label_count_report = _label_counts(label_ids, id_to_label)

    profile_path = final_dir / "outputs" / "reports" / "data_validation_profile.json"
    if profile_path.exists():
        p1_feature_stats = load_p1_feature_statistics(profile_path)
    else:
        p1_feature_stats = {}
        warnings.append("P1 feature profile missing; scaling status uses fallback heuristic")
    scaling_status = check_scaling_status(p1_feature_stats, feature_names)
    force_scaling = bool(scaling_config.get("force_scaling", False))
    skip_if_scaled = bool(scaling_config.get("skip_scaling_if_already_scaled", True))
    scaling_applied = scaling_status.status == "raw_unscaled" or force_scaling
    if scaling_status.status == "possibly_scaled" and skip_if_scaled and not force_scaling:
        scaling_applied = False
        warnings.append("scaling status is possibly_scaled; RobustScaler was skipped")

    l1_rows, l1_sampling, l1_sampling_errors = _build_l1_sample(
        label_ids,
        benign_label_id=benign_label_id,
        attack_samples_per_class=int(l1_config["attack_samples_per_class"]),
        random_seed=random_seed,
    )
    errors.extend(l1_sampling_errors)
    l1_labels = label_ids[l1_rows]
    l1_before_by_label = label_count_report
    l1_after_by_label = _label_counts(l1_labels, id_to_label)
    l1_sampling_report = {
        **l1_sampling,
        "count_before_sampling_by_label_id": l1_before_by_label,
        "count_after_sampling_by_label_id": l1_after_by_label,
        "selected_attack_classes": [
            {"label_id": label_id, "label_name": id_to_label[label_id]}
            for label_id in l1_sampling["selected_attack_label_ids"]
        ],
    }

    l1_split = stratified_train_val_test_split(
        l1_rows,
        l1_labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )
    l1_global_split = assign_to_global(label_ids.size, l1_rows, l1_split.assignments)
    l1_anti_leakage = anti_leakage_report(l1_global_split)

    l2_rows = np.flatnonzero(label_ids != benign_label_id).astype(np.int64)
    l2_labels = label_ids[l2_rows]
    l2_family_ids = lookups["family_id"][l2_labels]
    if np.any(l2_family_ids < 0):
        errors.append("L2 contains attack labels without family_id mapping")
    l2_split = stratified_train_val_test_split(
        l2_rows,
        l2_family_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )
    l2_global_split = assign_to_global(label_ids.size, l2_rows, l2_split.assignments)
    l2_anti_leakage = anti_leakage_report(l2_global_split)

    if errors:
        return _initial_failure_run(repo_root, reports_dir, errors, warnings)

    l1_paths = _write_preprocessed_splits(
        parquet_path,
        l1_dir,
        feature_names,
        label_column,
        l1_global_split,
        lookups,
        include_binary=True,
    )
    l2_paths = _write_preprocessed_splits(
        parquet_path,
        l2_dir,
        feature_names,
        label_column,
        l2_global_split,
        lookups,
        include_binary=False,
    )

    l1_scaler_path = scalers_dir / "l1_binary_robust_scaler.pkl"
    l2_scaler_path = scalers_dir / "l2_family_robust_scaler.pkl"
    l1_scaler_fit: RobustScalerFit | None = None
    l2_scaler_fit: RobustScalerFit | None = None

    if scaling_applied:
        l1_scaler_fit = fit_robust_scaler_from_parquet(l1_paths["train"], feature_names)
        save_scaler(l1_scaler_fit.scaler, l1_scaler_path)
        l2_scaler_fit = fit_robust_scaler_from_parquet(l2_paths["train"], feature_names)
        save_scaler(l2_scaler_fit.scaler, l2_scaler_path)

    l1_npz_infos = {
        split_name: transform_parquet_to_npz(
            parquet_path=path,
            output_path=l1_dir / f"{split_name}_scaled.npz",
            feature_names=feature_names,
            target_column="binary_label",
            target_output_name="y_binary",
            scaler=l1_scaler_fit.scaler if l1_scaler_fit else None,
            apply_scaling=scaling_applied,
        )
        for split_name, path in l1_paths.items()
    }
    l2_npz_infos = {
        split_name: transform_parquet_to_npz(
            parquet_path=path,
            output_path=l2_dir / f"{split_name}_scaled.npz",
            feature_names=feature_names,
            target_column="family_id",
            target_output_name="y_family",
            scaler=l2_scaler_fit.scaler if l2_scaler_fit else None,
            apply_scaling=scaling_applied,
        )
        for split_name, path in l2_paths.items()
    }

    family_labels = {family_id: name for name, family_id in FAMILY_ID_MAP.items()}
    binary_labels = {0: "normal", 1: "attack"}
    l1_binary_distribution_full = _count_by_binary(label_ids, benign_label_id)
    l1_binary_distribution_after = _count_by_binary(l1_labels, benign_label_id)
    l1_split_binary_distribution = split_counts_by_key(
        l1_global_split,
        lookups["binary_label"][label_ids],
        labels=binary_labels,
    )
    l1_split_label_distribution = split_counts_by_key(
        l1_global_split,
        label_ids,
        labels=id_to_label,
    )
    l2_family_distribution = _family_distribution(
        l2_labels,
        lookups["family_id"],
        family_labels,
    )
    l2_split_family_distribution = split_counts_by_key(
        l2_global_split,
        lookups["family_id"][label_ids],
        labels=family_labels,
    )
    l2_split_label_distribution = split_counts_by_key(
        l2_global_split,
        label_ids,
        labels=id_to_label,
    )

    l1_sampling_path = l1_dir / "sampling_report.json"
    _write_json(l1_sampling_path, l1_sampling_report)
    generated_files.append(_rel(l1_sampling_path, repo_root))

    l2_distribution_report = {
        "attack_only": True,
        "sampling": False,
        "total_count": int(l2_rows.size),
        "distribution_by_family_name": l2_family_distribution,
        "distribution_by_original_label_id": {
            key: value
            for key, value in label_count_report.items()
            if int(key) != benign_label_id
        },
        "family_id_mapping": FAMILY_ID_MAP,
    }
    l2_distribution_path = l2_dir / "distribution_report.json"
    _write_json(l2_distribution_path, l2_distribution_report)
    generated_files.append(_rel(l2_distribution_path, repo_root))

    l2_family_mapping_path = l2_dir / "family_mapping.json"
    _write_json(
        l2_family_mapping_path,
        {
            "family_to_id": FAMILY_ID_MAP,
            "id_to_family": {str(value): key for key, value in FAMILY_ID_MAP.items()},
        },
    )
    generated_files.append(_rel(l2_family_mapping_path, repo_root))

    source_p1_artifacts = {
        key: _rel(_repo_path(repo_root, inputs[key]), repo_root)
        for key in [
            "feature_names_path",
            "label_mapping_path",
            "id_to_label_path",
            "label_to_binary_path",
            "label_to_family_path",
        ]
    }

    l1_manifest = {
        "created_at": _created_at(),
        "source_dataset": _rel(parquet_path, repo_root),
        "source_p1_artifacts": source_p1_artifacts,
        "row_counts": {
            "total": int(l1_rows.size),
            "normal": int(np.sum(l1_labels == benign_label_id)),
            "attack": int(np.sum(l1_labels != benign_label_id)),
            "by_binary": l1_binary_distribution_after,
            "by_label_id": l1_after_by_label,
        },
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "actual_split_counts": l1_split.counts,
        "split_distribution_by_binary": l1_split_binary_distribution,
        "split_distribution_by_label": l1_split_label_distribution,
        "feature_count": len(feature_names),
        "label_columns": ["label_id", "label_name", "binary_label", "binary_name"],
        "scaler_path": _rel(l1_scaler_path, repo_root) if scaling_applied else None,
        "scaling_status": scaling_status.status,
        "scaling_applied": scaling_applied,
        "scaling_skipped": not scaling_applied,
        "scaling_train_only": True,
        "anti_leakage_id": split_config["anti_leakage_id"],
        "anti_leakage_result": l1_anti_leakage,
        "output_files": {
            "parquet": _split_file_infos(l1_paths, repo_root),
            "npz": {
                split_name: {
                    **info,
                    "path": _rel(Path(info["path"]), repo_root)
                    if Path(info["path"]).is_absolute()
                    else _rel(l1_dir / f"{split_name}_scaled.npz", repo_root),
                }
                for split_name, info in l1_npz_infos.items()
            },
        },
        "npz_shapes": {
            split_name: info["arrays"] for split_name, info in l1_npz_infos.items()
        },
        "random_seed": random_seed,
        "file_sizes": {
            "parquet": _split_file_infos(l1_paths, repo_root),
            "npz": {
                split_name: _file_info(l1_dir / f"{split_name}_scaled.npz", repo_root)
                for split_name in l1_paths
            },
            "scaler": _file_info(l1_scaler_path, repo_root) if scaling_applied else None,
        },
    }

    l2_manifest = {
        "created_at": _created_at(),
        "source_dataset": _rel(parquet_path, repo_root),
        "source_p1_artifacts": source_p1_artifacts,
        "row_counts": {
            "total": int(l2_rows.size),
            "attack_only": True,
            "by_family": l2_family_distribution,
            "by_label_id": l2_distribution_report["distribution_by_original_label_id"],
        },
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "actual_split_counts": l2_split.counts,
        "split_distribution_by_family": l2_split_family_distribution,
        "split_distribution_by_label": l2_split_label_distribution,
        "feature_count": len(feature_names),
        "label_columns": ["label_id", "label_name", "family_name", "family_id"],
        "scaler_path": _rel(l2_scaler_path, repo_root) if scaling_applied else None,
        "scaling_status": scaling_status.status,
        "scaling_applied": scaling_applied,
        "scaling_skipped": not scaling_applied,
        "scaling_train_only": True,
        "anti_leakage_id": split_config["anti_leakage_id"],
        "anti_leakage_result": l2_anti_leakage,
        "output_files": {
            "parquet": _split_file_infos(l2_paths, repo_root),
            "npz": {
                split_name: {
                    **info,
                    "path": _rel(l2_dir / f"{split_name}_scaled.npz", repo_root),
                }
                for split_name, info in l2_npz_infos.items()
            },
        },
        "npz_shapes": {
            split_name: info["arrays"] for split_name, info in l2_npz_infos.items()
        },
        "random_seed": random_seed,
        "family_id_mapping": FAMILY_ID_MAP,
        "file_sizes": {
            "parquet": _split_file_infos(l2_paths, repo_root),
            "npz": {
                split_name: _file_info(l2_dir / f"{split_name}_scaled.npz", repo_root)
                for split_name in l2_paths
            },
            "scaler": _file_info(l2_scaler_path, repo_root) if scaling_applied else None,
        },
    }

    l1_manifest_path = l1_dir / "manifest.json"
    l2_manifest_path = l2_dir / "manifest.json"
    _write_json(l1_manifest_path, l1_manifest)
    _write_json(l2_manifest_path, l2_manifest)
    generated_files.extend([_rel(l1_manifest_path, repo_root), _rel(l2_manifest_path, repo_root)])

    figures = {
        "scaling_status": figures_dir / "01_scaling_status_check.png",
        "l1_before_after": figures_dir / "02_l1_binary_distribution_before_after.png",
        "l1_attack_sampling": figures_dir / "03_l1_attack_sampling_per_class.png",
        "l1_split": figures_dir / "04_l1_train_val_test_distribution.png",
        "l2_family": figures_dir / "05_l2_family_distribution_full_attack_only.png",
        "l2_split": figures_dir / "06_l2_train_val_test_distribution.png",
        "l1_scaling": figures_dir / "07_feature_scaling_before_after_l1.png",
        "l2_scaling": figures_dir / "08_feature_scaling_before_after_l2.png",
        "pipeline": figures_dir / "09_preprocessing_pipeline_l1_l2.png",
    }
    _generate_scaling_status_figure(scaling_status, figures["scaling_status"])
    _generate_binary_before_after_figure(
        l1_binary_distribution_full,
        l1_binary_distribution_after,
        figures["l1_before_after"],
    )
    attack_labels = [
        id_to_label[int(label_id)]
        for label_id in sorted(np.unique(l1_labels).tolist())
        if int(label_id) != benign_label_id
    ]
    attack_counts_after = [
        l1_after_by_label[str(label_id)]["count"]
        for label_id in sorted(np.unique(l1_labels).tolist())
        if int(label_id) != benign_label_id
    ]
    _generate_bar(
        attack_labels,
        attack_counts_after,
        figures["l1_attack_sampling"],
        "L1 attack sampling per original class",
        "Rows sampled",
        horizontal=True,
        color="#16A34A",
    )
    _generate_grouped_split_figure(
        l1_split_binary_distribution,
        figures["l1_split"],
        "L1 train/val/test binary distribution",
    )
    _generate_bar(
        list(l2_family_distribution.keys()),
        [entry["count"] for entry in l2_family_distribution.values()],
        figures["l2_family"],
        "L2 family distribution, attack-only full dataset",
        "Rows",
        horizontal=True,
        color="#0F766E",
    )
    _generate_grouped_split_figure(
        l2_split_family_distribution,
        figures["l2_split"],
        "L2 train/val/test family distribution",
    )
    _generate_scaling_before_after_figure(
        l1_scaler_fit, figures["l1_scaling"], "L1 feature scaling before/after"
    )
    _generate_scaling_before_after_figure(
        l2_scaler_fit, figures["l2_scaling"], "L2 feature scaling before/after"
    )
    _generate_pipeline_figure(figures["pipeline"])
    generated_files.extend([_rel(path, repo_root) for path in figures.values()])

    generated_artifacts = [
        _rel(l1_manifest_path, repo_root),
        _rel(l1_sampling_path, repo_root),
        _rel(l2_manifest_path, repo_root),
        _rel(l2_distribution_path, repo_root),
        _rel(l2_family_mapping_path, repo_root),
    ]
    if scaling_applied:
        generated_artifacts.extend(
            [_rel(l1_scaler_path, repo_root), _rel(l2_scaler_path, repo_root)]
        )

    criteria = {
        "p1_validated_detected": p1_check["p1_accepted"],
        "parquet_source_found": parquet_path.exists(),
        "features_28_confirmed": len(feature_names) == int(schema["expected_num_features"]),
        "scaling_status_check_executed": scaling_status.status in {"raw_unscaled", "possibly_scaled"},
        "l1_total_630000": int(l1_rows.size) == int(l1_config["expected_total_count"]),
        "l1_normal_300000": int(np.sum(l1_labels == benign_label_id)) == int(l1_config["normal_count"]),
        "l1_attack_330000": int(np.sum(l1_labels != benign_label_id))
        == int(l1_config["expected_attack_count"]),
        "l1_attack_sampling_10000_each": all(
            entry["count"] == int(l1_config["attack_samples_per_class"])
            for key, entry in l1_after_by_label.items()
            if int(key) != benign_label_id
        ),
        "l1_split_generated": all(path.exists() for path in l1_paths.values()),
        "l1_anti_leakage_valid": l1_anti_leakage["anti_leakage_valid"],
        "l1_npz_generated": all((l1_dir / f"{split}_scaled.npz").exists() for split in l1_paths),
        "l2_attack_only_no_sampling": int(l2_rows.size) == int(l2_config["expected_total_count"]),
        "l2_excludes_benign": not bool(np.any(l2_labels == benign_label_id)),
        "l2_eight_families": set(l2_family_distribution) == set(l2_config["expected_families"]),
        "l2_split_generated": all(path.exists() for path in l2_paths.values()),
        "l2_anti_leakage_valid": l2_anti_leakage["anti_leakage_valid"],
        "l2_npz_generated": all((l2_dir / f"{split}_scaled.npz").exists() for split in l2_paths),
        "manifests_generated": l1_manifest_path.exists() and l2_manifest_path.exists(),
        "reports_generated": True,
        "figures_9_generated": all(path.exists() for path in figures.values()),
    }
    accepted = all(criteria.values()) and not errors

    summary = {
        "accepted": accepted,
        "created_at": _created_at(),
        "inputs": {
            "parquet_path": _rel(parquet_path, repo_root),
            "p1_artifacts_dir": _rel(final_dir / "outputs" / "artifacts", repo_root),
        },
        "schema": {
            "feature_count": len(feature_names),
            "label_column": label_column,
            "num_classes": len(label_mapping),
        },
        "scaling": {
            "status": scaling_status.status,
            "justification": scaling_status.justification,
            "indicators": scaling_status.indicators,
            "applied": scaling_applied,
            "train_only": True,
        },
        "l1_binary": {
            "total_count": int(l1_rows.size),
            "normal_count": int(np.sum(l1_labels == benign_label_id)),
            "attack_count": int(np.sum(l1_labels != benign_label_id)),
            "attack_samples_per_class": int(l1_config["attack_samples_per_class"]),
            "split_counts": l1_split.counts,
            "anti_leakage_result": l1_anti_leakage,
            "scaler_path": _rel(l1_scaler_path, repo_root) if scaling_applied else None,
            "npz_files": {
                split_name: _rel(l1_dir / f"{split_name}_scaled.npz", repo_root)
                for split_name in l1_paths
            },
        },
        "l2_family": {
            "total_count": int(l2_rows.size),
            "families": sorted(l2_family_distribution.keys()),
            "sampling": False,
            "split_counts": l2_split.counts,
            "anti_leakage_result": l2_anti_leakage,
            "scaler_path": _rel(l2_scaler_path, repo_root) if scaling_applied else None,
            "npz_files": {
                split_name: _rel(l2_dir / f"{split_name}_scaled.npz", repo_root)
                for split_name in l2_paths
            },
        },
        "generated_artifacts": generated_artifacts,
        "figures": [_rel(path, repo_root) for path in figures.values()],
        "criteria": criteria,
        "warnings": warnings,
        "errors": errors,
    }
    profile = {
        "p1_check": p1_check,
        "parquet_metadata": parquet_metadata,
        "label_distribution_full": label_count_report,
        "scaling": summary["scaling"],
        "l1_binary": {
            "before_sampling": l1_binary_distribution_full,
            "after_sampling": l1_binary_distribution_after,
            "sampling_report": l1_sampling_report,
            "split_by_binary": l1_split_binary_distribution,
            "split_by_label": l1_split_label_distribution,
            "scaler_feature_stats": l1_scaler_fit.feature_stats if l1_scaler_fit else None,
            "manifest": l1_manifest,
        },
        "l2_family": {
            "distribution_report": l2_distribution_report,
            "split_by_family": l2_split_family_distribution,
            "split_by_label": l2_split_label_distribution,
            "scaler_feature_stats": l2_scaler_fit.feature_stats if l2_scaler_fit else None,
            "manifest": l2_manifest,
        },
        "warnings": warnings,
        "errors": errors,
    }

    summary_path = reports_dir / "preprocessing_summary.json"
    profile_report_path = reports_dir / "preprocessing_profile.json"
    _write_json(summary_path, summary)
    _write_json(profile_report_path, profile)
    generated_files.extend([_rel(summary_path, repo_root), _rel(profile_report_path, repo_root)])

    markdown_path = final_dir / "docs" / "02_preprocessing.md"
    _write_markdown_report(markdown_path, summary, profile)
    generated_files.append(_rel(markdown_path, repo_root))
    generated_files.extend(
        [_rel(path, repo_root) for path in [*l1_paths.values(), *l2_paths.values()]]
    )
    generated_files.extend(
        [
            _rel(l1_dir / f"{split_name}_scaled.npz", repo_root)
            for split_name in l1_paths
        ]
    )
    generated_files.extend(
        [
            _rel(l2_dir / f"{split_name}_scaled.npz", repo_root)
            for split_name in l2_paths
        ]
    )
    if scaling_applied:
        generated_files.extend([_rel(l1_scaler_path, repo_root), _rel(l2_scaler_path, repo_root)])

    return PreprocessingRun(
        summary=summary,
        profile=profile,
        errors=errors,
        warnings=warnings,
        generated_files=sorted(set(generated_files)),
    )
