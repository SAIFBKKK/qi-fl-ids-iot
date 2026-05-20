"""Build P12 final ablation and evaluation reports.

This script consolidates existing evidence only. It does not train, launch
Flower, or touch local run artifacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path.cwd()
BASE = ROOT / "experiments/qi-fl-ids-iot-final"
REPORTS = BASE / "outputs/reports"
FIGURES = BASE / "outputs/figures/p12_ablation"
DOCS = BASE / "docs"

SOURCES = {
    "p8_qga_ablation_summary": REPORTS / "p8_qga_ablation_summary.csv",
    "p8b_qga_l2_ablation_summary": REPORTS / "p8b_qga_l2_ablation_summary.csv",
    "p9_qifa_ablation_summary": REPORTS / "p9_qifa_ablation_summary.csv",
    "p10_robustness_full_summary": REPORTS / "p10_robustness_full_summary.csv",
    "p10_robustness_clean_vs_poisoned": REPORTS / "p10_robustness_clean_vs_poisoned.csv",
    "p11_fedtn_mps_summary": REPORTS / "p11_fedtn_mps_summary.csv",
    "p5_grid_summary": REPORTS / "p5_grid_summary.csv",
    "p7_multitier_summary": REPORTS / "p7_multitier_summary.csv",
    "p10_global_comparison": REPORTS / "p10_global_comparison.csv",
}

FIELDS = [
    "phase",
    "method",
    "task",
    "features_count",
    "macro_f1",
    "weighted_f1",
    "attack_recall",
    "fpr",
    "accuracy",
    "model_size_bytes",
    "bandwidth_total_bytes",
    "compression_ratio",
    "bandwidth_reduction_ratio",
    "robustness_score",
    "runtime",
    "true_flower_runtime",
    "result_type",
    "evidence_level",
    "accepted",
    "source",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_bool(value: Any) -> str:
    return str(value).lower() in {"true", "1", "yes"}


def normalize_task(value: str) -> str:
    lower = str(value).lower()
    if "l2" in lower:
        return "l2_family"
    if "robust" in lower:
        return "l1_binary_robustness"
    if "compression" in lower:
        return "l1_binary_compression"
    return "l1_binary"


def make_row(**kwargs: Any) -> dict[str, Any]:
    return {field: kwargs.get(field, "") for field in FIELDS}


def load_global_comparison() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_row in read_csv(SOURCES["p10_global_comparison"]):
        rows.append(
            make_row(
                phase=source_row.get("phase", ""),
                method=source_row.get("method", ""),
                task=normalize_task(source_row.get("task", "")),
                features_count=source_row.get("features_count", ""),
                macro_f1=source_row.get("macro_f1", ""),
                weighted_f1=source_row.get("weighted_f1", ""),
                attack_recall=source_row.get("attack_recall", ""),
                fpr=source_row.get("fpr", ""),
                accuracy=source_row.get("accuracy", ""),
                model_size_bytes=source_row.get("model_size_bytes", ""),
                bandwidth_total_bytes=source_row.get("bandwidth_total_bytes", ""),
                runtime=source_row.get("runtime", ""),
                true_flower_runtime=source_row.get("true_flower_runtime", ""),
                result_type="measured",
                evidence_level="final_full",
                accepted=source_row.get("accepted", ""),
                source=source_row.get("source", "outputs/reports/p10_global_comparison.csv"),
            )
        )
    return rows


def load_robustness_rows() -> list[dict[str, Any]]:
    labels = {
        "fedavg": "P10 FedAvg poisoned",
        "fedavg_qga": "P10 FedAvg + QGA poisoned",
        "qifa": "P10 QIFA poisoned",
        "qifa_qga": "P10 QIFA + QGA poisoned",
    }
    rows: list[dict[str, Any]] = []
    for source_row in read_csv(SOURCES["p10_robustness_full_summary"]):
        method = source_row.get("method", "")
        rows.append(
            make_row(
                phase="P10",
                method=labels.get(method, f"P10 {method} poisoned"),
                task="l1_binary_robustness",
                features_count="12" if method.endswith("_qga") else "28",
                macro_f1=source_row.get("macro_f1", ""),
                attack_recall=source_row.get("attack_recall", ""),
                fpr=source_row.get("fpr", ""),
                accuracy=source_row.get("accuracy", ""),
                robustness_score=source_row.get("robustness_score", ""),
                runtime="in_process_robustness",
                true_flower_runtime="False",
                result_type="measured",
                evidence_level="final_full",
                accepted=source_row.get("accepted", ""),
                source="outputs/reports/p10_robustness_full_summary.csv",
            )
        )
    return rows


def load_compression_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_row in read_csv(SOURCES["p11_fedtn_mps_summary"]):
        rank = source_row.get("rank", "")
        base_model = source_row.get("base_model", "")
        rows.append(
            make_row(
                phase="P11",
                method=f"P11 FedTN/MPS {base_model} rank {rank}",
                task="l1_binary_compression",
                features_count="12",
                model_size_bytes=source_row.get("compressed_model_size_bytes", ""),
                bandwidth_total_bytes=source_row.get("compressed_bandwidth_total_bytes", ""),
                compression_ratio=source_row.get("compression_ratio", ""),
                bandwidth_reduction_ratio=source_row.get("bandwidth_reduction_ratio", ""),
                runtime="dry_run",
                true_flower_runtime="False",
                result_type="dry_run",
                evidence_level="dry_run",
                accepted=source_row.get("accepted", ""),
                source="outputs/reports/p11_fedtn_mps_summary.csv",
            )
        )
    return rows


def build_audit() -> dict[str, Any]:
    audit = {"found": {}, "missing": [], "columns": {}, "notes": []}
    for name, path in SOURCES.items():
        exists = path.exists()
        audit["found"][name] = str(path) if exists else ""
        if not exists:
            audit["missing"].append(str(path))
            continue
        rows = read_csv(path)
        audit["columns"][name] = list(rows[0].keys()) if rows else []
    audit["notes"] = [
        "P11 FedTN/MPS values are structural dry-run estimates, not measured Macro-F1/Recall/FPR.",
        "P10 robustness rows are measured full scenarios under label_flip poison_rate=0.2.",
        "P8/P9 L1 rows are measured full Flower/FedAvg evidence from existing reports.",
    ]
    lines = ["# P12 Ablation Audit", "", "## Files Found"]
    for name, path in audit["found"].items():
        lines.append(f"- {name}: {'FOUND ' + path if path else 'MISSING'}")
    lines.extend(["", "## Missing Files"])
    if audit["missing"]:
        lines.extend(f"- {path}" for path in audit["missing"])
    else:
        lines.append("- None")
    lines.extend(["", "## Measured vs Estimated"])
    lines.extend(f"- {note}" for note in audit["notes"])
    lines.extend(["", "## Metrics Availability"])
    for name, columns in audit["columns"].items():
        lines.append(f"- {name}: {', '.join(columns) if columns else 'no rows'}")
    (REPORTS / "p12_ablation_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "l1": [row for row in rows if row["task"] == "l1_binary" and row["phase"] in {"P4", "P5", "P8", "P9", "P7"}],
        "l2": [row for row in rows if row["task"] == "l2_family"],
        "robustness": [row for row in rows if row["task"] == "l1_binary_robustness"],
        "compression": [row for row in rows if row["task"] == "l1_binary_compression"],
    }


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    header = "| phase | method | task | macro_f1 | attack_recall | fpr | bandwidth | result_type |\n|---|---|---|---:|---:|---:|---:|---|\n"
    lines = [
        f"| {row['phase']} | {row['method']} | {row['task']} | {row['macro_f1']} | {row['attack_recall']} | {row['fpr']} | {row['bandwidth_total_bytes']} | {row['result_type']} |"
        for row in rows
    ]
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def plot_bar(rows: list[dict[str, Any]], metric: str, filename: str, title: str, ylabel: str) -> Path:
    path = FIGURES / filename
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [row["method"] for row in rows]
    values = [to_float(row.get(metric), 0.0) or 0.0 for row in rows]
    ax.bar(range(len(rows)), values, color="#2563EB")
    ax.set_xticks(range(len(rows)), labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_tradeoff(rows: list[dict[str, Any]]) -> Path:
    path = FIGURES / "p12_l1_attack_recall_fpr_tradeoff.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        recall = to_float(row.get("attack_recall"))
        fpr = to_float(row.get("fpr"))
        if recall is None or fpr is None:
            continue
        ax.scatter(fpr, recall, s=80)
        ax.text(fpr, recall, row["method"].replace(" ", "\n"), fontsize=8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("Attack recall")
    ax.set_title("L1 attack recall / FPR tradeoff")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_ranking_table(rows: list[dict[str, Any]]) -> Path:
    path = FIGURES / "p12_final_method_ranking_table.png"
    ranked = sorted(
        [row for row in rows if row["result_type"] == "measured" and to_float(row.get("macro_f1")) is not None],
        key=lambda row: to_float(row.get("macro_f1"), 0.0) or 0.0,
        reverse=True,
    )[:8]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")
    table_data = [[row["method"], row["task"], row["macro_f1"], row["attack_recall"], row["fpr"]] for row in ranked]
    table = ax.table(cellText=table_data, colLabels=["method", "task", "macro_f1", "attack_recall", "fpr"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    ax.set_title("Final measured method ranking")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_best_objectives() -> Path:
    path = FIGURES / "p12_best_model_by_objective.png"
    objectives = [
        ("Production L1", "P8 FedAvg + QGA"),
        ("Attack recall", "P9 QIFA + QGA"),
        ("Low FPR", "P9 QIFA"),
        ("Robustness", "P10 QIFA + QGA"),
        ("Compression", "P11 FedTN/MPS rank 8"),
        ("L2 experimental", "P8-b QGA L2"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    table = ax.table(cellText=objectives, colLabels=["objective", "recommended method"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title("Best model by objective")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def generate_figures(groups: dict[str, list[dict[str, Any]]], all_rows: list[dict[str, Any]]) -> list[str]:
    figures = [
        plot_bar(groups["l1"], "macro_f1", "p12_l1_macro_f1_comparison.png", "L1 Macro-F1 comparison", "Macro-F1"),
        plot_tradeoff(groups["l1"]),
        plot_bar(groups["l1"], "bandwidth_total_bytes", "p12_l1_bandwidth_comparison.png", "L1 bandwidth comparison", "bytes"),
        plot_bar([row for row in groups["l1"] if row["phase"] in {"P5", "P8", "P9"}], "macro_f1", "p12_qga_qifa_contribution_summary.png", "QGA/QIFA contribution summary", "Macro-F1"),
        plot_bar(groups["l2"], "macro_f1", "p12_l2_p6_vs_p8b.png", "L2 P6 vs P8-b", "Macro-F1"),
        plot_bar(groups["robustness"], "robustness_score", "p12_robustness_under_poisoning.png", "P10 robustness under poisoning", "robustness score"),
        plot_bar(groups["compression"], "model_size_bytes", "p12_compression_size_reduction.png", "P11 compressed model size", "bytes"),
        plot_ranking_table(all_rows),
        plot_best_objectives(),
    ]
    return [str(path) for path in figures]


def write_findings(rows: list[dict[str, Any]], figures: list[str]) -> None:
    content = f"""# P12 Final Findings

## Evidence Policy

P12 consolidates existing reports only. No full FL, Flower, robustness grid, Docker, dashboard, P13, or training run is launched.

Measured rows are final full evidence. P11 FedTN/MPS rows are dry-run structural estimates and must not be interpreted as measured Macro-F1, recall, or FPR.

## Final Objective Ranking

- Best production L1 compromise: P8 FedAvg + QGA.
- Best attack recall: P9 QIFA + QGA.
- Best FPR: P9 QIFA.
- Best poisoning robustness: P10 QIFA + QGA.
- Best structural compression: P11 FedTN/MPS rank 8.
- L2 experimental direction: P8-b QGA L2.

## Global Summary

Rows consolidated: {len(rows)}.

## Figures

{chr(10).join(f'- {figure}' for figure in figures)}

## Limitations

- P11 has no checkpoint-based measured Macro-F1/Recall/FPR in the current evidence pack.
- Multi-tier and L2 rows are experimental and not dashboard production candidates.
- Dashboard recommendation remains L1 production, with P8 FedAvg + QGA as deployment-oriented reference and P9/P10 as robustness evidence.
"""
    (REPORTS / "p12_final_findings.md").write_text(content, encoding="utf-8")


def write_doc() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "12_ablation_and_evaluation_reports.md").write_text(
        """# P12 - Ablation and Evaluation Reports

## Objective

P12 consolidates the final evidence across L1, L2, robustness, and compression. It distinguishes measured metrics from dry-run or estimated structural values.

## Methods Compared

- P4 Centralized L1
- P5 FedAvg L1
- P8 FedAvg + QGA L1
- P9 QIFA L1
- P9 QIFA + QGA L1
- P6 L2 Flower baseline
- P8-b L2 FedAvg + QGA Flower
- P7 HeteroFL L1/L2 best rows
- P10 poisoned robustness rows
- P11 FedTN/MPS rank dry-run rows

## Results L1

P8 FedAvg + QGA is the best production L1 compromise because it preserves strong Macro-F1 while reducing features and bandwidth. P9 QIFA + QGA gives the strongest attack recall.

## Results L2

L2 remains experimental. P8-b QGA L2 improves the L2 Flower baseline in Macro-F1 and model size, but it is not a dashboard deployment target.

## Robustness

P10 shows QIFA + QGA is the strongest method under label flipping, with the best robustness score and attack recall.

## Compression

P11 FedTN/MPS rank 8 is a structural dry-run. It reduces model size and estimated bandwidth, but no measured Macro-F1/Recall/FPR is attached without a checkpoint evaluation.

## Final Ranking by Objective

- Best production L1: P8 FedAvg + QGA.
- Best attack recall: P9 QIFA + QGA.
- Best FPR: P9 QIFA.
- Best poisoning robustness: P10 QIFA + QGA.
- Best structural compression: P11 FedTN/MPS rank 8.
- L2 experimental: P8-b QGA L2.

## Recommendation for P13 Dashboard

Use L1 production artifacts and expose QGA/FedAvg metrics as the stable dashboard baseline. Keep QIFA, robustness, L2, and FedTN/MPS as research evidence panels or appendix material unless a later deployment phase hardens them.
""",
        encoding="utf-8",
    )


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    rows = load_global_comparison() + load_robustness_rows() + load_compression_rows()
    groups = split_rows(rows)
    write_csv(REPORTS / "p12_global_ablation_summary.csv", rows)
    write_json(
        REPORTS / "p12_global_ablation_summary.json",
        {"rows": rows, "row_count": len(rows), "audit": audit, "warnings": audit["missing"]},
    )
    write_markdown_table(REPORTS / "p12_global_ablation_table.md", rows)
    write_csv(REPORTS / "p12_l1_ablation_summary.csv", groups["l1"])
    write_csv(REPORTS / "p12_l2_ablation_summary.csv", groups["l2"])
    write_csv(REPORTS / "p12_robustness_summary.csv", groups["robustness"])
    write_csv(REPORTS / "p12_compression_summary.csv", groups["compression"])
    figures = generate_figures(groups, rows)
    write_findings(rows, figures)
    write_json(
        REPORTS / "p12_evaluation_manifest.json",
        {
            "reports": [
                "p12_global_ablation_summary.csv",
                "p12_global_ablation_summary.json",
                "p12_global_ablation_table.md",
                "p12_l1_ablation_summary.csv",
                "p12_l2_ablation_summary.csv",
                "p12_robustness_summary.csv",
                "p12_compression_summary.csv",
                "p12_final_findings.md",
            ],
            "figures": figures,
            "result_type_policy": {
                "measured": "Full measured evidence from existing reports.",
                "estimated": "Derived non-training estimate.",
                "dry_run": "Structural dry-run without checkpoint metrics.",
                "structural": "Architecture or size-only value.",
            },
        },
    )
    write_doc()
    print(f"P12 final ablation report built rows={len(rows)} figures={len(figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
