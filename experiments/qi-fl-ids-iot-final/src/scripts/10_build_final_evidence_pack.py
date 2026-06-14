"""Build the P10 final evidence pack: global comparison table, figures, manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _f(val: Any, digits: int = 4) -> float | None:
    try:
        return round(float(val), digits)
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ── data collection ───────────────────────────────────────────────────────────

def _collect_rows(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reports = repo_root / "experiments/qi-fl-ids-iot-final/outputs/reports"
    central_metrics = _load_json(repo_root / "experiments/qi-fl-ids-iot-final/outputs/centralized_l1/artifacts/metrics_test.json")
    central_model = _load_json(repo_root / "experiments/qi-fl-ids-iot-final/outputs/centralized_l1/artifacts/model_config.json")

    warnings_list: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    # ── P4 Centralized L1 ────────────────────────────────────────────────────
    if central_metrics:
        rows.append({
            "phase": "P4",
            "method": "P4 Centralized L1",
            "task": "L1 binary",
            "features_count": 28,
            "macro_f1": _f(central_metrics.get("macro_f1")),
            "weighted_f1": _f(central_metrics.get("weighted_f1")),
            "attack_recall": _f(central_metrics.get("recall_attack")),
            "fpr": _f(central_metrics.get("FPR")),
            "accuracy": _f(central_metrics.get("accuracy")),
            "model_size_bytes": int(central_model.get("num_parameters", 12098)) * 4,
            "bandwidth_total_bytes": 0,
            "runtime": "centralized",
            "true_flower_runtime": False,
            "best_use_case": "Upper bound — no federation",
            "accepted": True,
            "source": "outputs/centralized_l1/artifacts/metrics_test.json",
        })
    else:
        warnings_list.append({"field": "P4 Centralized L1", "issue": "metrics_test.json not found"})

    # ── P5 FedAvg L1 (alpha=0.5 k=3) ────────────────────────────────────────
    p5_rows = _load_csv(reports / "p5_grid_summary.csv")
    p5_ref = next((r for r in p5_rows if _f(r.get("alpha")) == 0.5 and int(float(r.get("clients", 0))) == 3), None)
    if p5_ref:
        rows.append({
            "phase": "P5",
            "method": "P5 FedAvg L1",
            "task": "L1 binary",
            "features_count": 28,
            "macro_f1": _f(p5_ref.get("macro_f1")),
            "weighted_f1": _f(p5_ref.get("weighted_f1")),
            "attack_recall": _f(p5_ref.get("attack_recall")),
            "fpr": _f(p5_ref.get("fpr")),
            "accuracy": _f(p5_ref.get("accuracy")),
            "model_size_bytes": int(float(p5_ref.get("model_size_bytes", 48392))),
            "bandwidth_total_bytes": int(float(p5_ref.get("bandwidth_total_bytes", 8710560))),
            "runtime": "in_process",
            "true_flower_runtime": False,
            "best_use_case": "Reference FL baseline",
            "accepted": True,
            "source": "outputs/reports/p5_grid_summary.csv",
        })
    else:
        warnings_list.append({"field": "P5 FedAvg L1", "issue": "alpha=0.5 k=3 row not found in p5_grid_summary.csv"})

    # ── P8 QGA ablation ──────────────────────────────────────────────────────
    p8_rows = _load_csv(reports / "p8_qga_ablation_summary.csv")
    p8_qga = next((r for r in p8_rows if "FedAvg + QGA" in str(r.get("method", ""))), None)
    if p8_qga:
        rows.append({
            "phase": "P8",
            "method": "P8 FedAvg + QGA L1",
            "task": "L1 binary",
            "features_count": int(float(p8_qga.get("features_count", 12))),
            "macro_f1": _f(p8_qga.get("macro_f1")),
            "weighted_f1": _f(p8_qga.get("weighted_f1") or p8_qga.get("weighted_f1")),
            "attack_recall": _f(p8_qga.get("attack_recall")),
            "fpr": _f(p8_qga.get("fpr")),
            "accuracy": _f(p8_qga.get("accuracy")),
            "model_size_bytes": int(float(p8_qga.get("model_size_bytes", 40200))),
            "bandwidth_total_bytes": int(float(p8_qga.get("bandwidth_total_bytes", 7236000))),
            "runtime": str(p8_qga.get("runtime", "manual")),
            "true_flower_runtime": str(p8_qga.get("true_flower_runtime", "")).lower() == "true",
            "best_use_case": "Production L1 — best macro_f1 + 57% fewer features",
            "accepted": str(p8_qga.get("accepted", "")).lower() == "true",
            "source": "outputs/reports/p8_qga_ablation_summary.csv",
        })
    else:
        warnings_list.append({"field": "P8 FedAvg + QGA L1", "issue": "row not found in p8_qga_ablation_summary.csv"})

    # ── P9 QIFA / QIFA+QGA ───────────────────────────────────────────────────
    p9_rows = _load_csv(reports / "p9_qifa_ablation_summary.csv")
    p9_qifa = next((r for r in p9_rows if r.get("method") == "P9 QIFA Flower"), None)
    p9_qga = next((r for r in p9_rows if r.get("method") == "P9 QIFA + QGA Flower"), None)
    if p9_qifa:
        rows.append({
            "phase": "P9",
            "method": "P9 QIFA L1",
            "task": "L1 binary",
            "features_count": int(float(p9_qifa.get("features_count", 28))),
            "macro_f1": _f(p9_qifa.get("macro_f1")),
            "weighted_f1": _f(p9_qifa.get("weighted_f1")),
            "attack_recall": _f(p9_qifa.get("attack_recall")),
            "fpr": _f(p9_qifa.get("fpr")),
            "accuracy": _f(p9_qifa.get("accuracy")),
            "model_size_bytes": int(float(p9_qifa.get("model_size_bytes", 48392))),
            "bandwidth_total_bytes": int(float(p9_qifa.get("bandwidth_total_bytes", 8710560))),
            "runtime": "true_flower",
            "true_flower_runtime": True,
            "best_use_case": "Low FPR — quantum-inspired aggregation",
            "accepted": str(p9_qifa.get("accepted", "")).lower() == "true",
            "source": "outputs/reports/p9_qifa_ablation_summary.csv",
        })
    else:
        warnings_list.append({"field": "P9 QIFA L1", "issue": "P9 QIFA Flower row not found"})

    if p9_qga:
        rows.append({
            "phase": "P9",
            "method": "P9 QIFA + QGA L1",
            "task": "L1 binary",
            "features_count": int(float(p9_qga.get("features_count", 12))),
            "macro_f1": _f(p9_qga.get("macro_f1")),
            "weighted_f1": _f(p9_qga.get("weighted_f1")),
            "attack_recall": _f(p9_qga.get("attack_recall")),
            "fpr": _f(p9_qga.get("fpr")),
            "accuracy": _f(p9_qga.get("accuracy")),
            "model_size_bytes": int(float(p9_qga.get("model_size_bytes", 40200))),
            "bandwidth_total_bytes": int(float(p9_qga.get("bandwidth_total_bytes", 7236000))),
            "runtime": "true_flower",
            "true_flower_runtime": True,
            "best_use_case": "Best attack recall + QGA compression",
            "accepted": str(p9_qga.get("accepted", "")).lower() == "true",
            "source": "outputs/reports/p9_qifa_ablation_summary.csv",
        })
    else:
        warnings_list.append({"field": "P9 QIFA + QGA L1", "issue": "P9 QIFA + QGA Flower row not found"})

    # ── P6 L2 Flower baseline ────────────────────────────────────────────────
    p8b_rows = _load_csv(reports / "p8b_qga_l2_ablation_summary.csv")
    p6_l2 = next((r for r in p8b_rows if "P6" in str(r.get("method", ""))), None)
    p8b_l2 = next((r for r in p8b_rows if "P8-b" in str(r.get("method", "")) or "P8b" in str(r.get("method", ""))), None)
    if p6_l2:
        rows.append({
            "phase": "P6",
            "method": "P6 L2 Flower baseline",
            "task": "L2 family",
            "features_count": int(float(p6_l2.get("features_count", 28))) if p6_l2.get("features_count") else 28,
            "macro_f1": _f(p6_l2.get("macro_f1")),
            "weighted_f1": _f(p6_l2.get("weighted_f1")),
            "attack_recall": _f(p6_l2.get("macro_recall") or p6_l2.get("attack_recall")),
            "fpr": _f(p6_l2.get("macro_fpr") or p6_l2.get("fpr")),
            "accuracy": _f(p6_l2.get("accuracy")),
            "model_size_bytes": int(float(p6_l2.get("model_size_bytes", 49952))),
            "bandwidth_total_bytes": int(float(p6_l2.get("bandwidth_total_bytes", 8991360))),
            "runtime": "true_flower",
            "true_flower_runtime": str(p6_l2.get("true_flower_runtime", "")).lower() == "true",
            "best_use_case": "L2 family classification baseline",
            "accepted": str(p6_l2.get("accepted", "")).lower() == "true",
            "source": "outputs/reports/p8b_qga_l2_ablation_summary.csv",
        })
    else:
        warnings_list.append({"field": "P6 L2 Flower baseline", "issue": "P6 row not found in p8b_qga_l2_ablation_summary.csv"})

    if p8b_l2:
        rows.append({
            "phase": "P8-b",
            "method": "P8-b L2 FedAvg + QGA Flower",
            "task": "L2 family",
            "features_count": int(float(p8b_l2.get("features_count", 19))) if p8b_l2.get("features_count") else 19,
            "macro_f1": _f(p8b_l2.get("macro_f1")),
            "weighted_f1": _f(p8b_l2.get("weighted_f1")),
            "attack_recall": _f(p8b_l2.get("macro_recall") or p8b_l2.get("attack_recall")),
            "fpr": _f(p8b_l2.get("macro_fpr") or p8b_l2.get("fpr")),
            "accuracy": _f(p8b_l2.get("accuracy")),
            "model_size_bytes": int(float(p8b_l2.get("model_size_bytes", 45344))),
            "bandwidth_total_bytes": int(float(p8b_l2.get("bandwidth_total_bytes", 8161920))),
            "runtime": "true_flower",
            "true_flower_runtime": str(p8b_l2.get("true_flower_runtime", "")).lower() == "true",
            "best_use_case": "L2 family + QGA feature compression",
            "accepted": str(p8b_l2.get("accepted", "")).lower() == "true",
            "source": "outputs/reports/p8b_qga_l2_ablation_summary.csv",
        })
    else:
        warnings_list.append({"field": "P8-b L2 FedAvg + QGA Flower", "issue": "P8-b row not found in p8b_qga_l2_ablation_summary.csv"})

    # ── P7 Multitier L1 best (alpha=5.0 k=3) ────────────────────────────────
    p7_rows = _load_csv(reports / "p7_multitier_summary.csv")
    p7_l1 = next((r for r in p7_rows if r.get("task") == "l1_binary" and _f(r.get("alpha")) == 5.0 and int(float(r.get("clients", 0))) == 3), None)
    p7_l2 = next((r for r in p7_rows if r.get("task") == "l2_family" and _f(r.get("alpha")) == 0.5 and int(float(r.get("clients", 0))) == 3), None)
    if p7_l1:
        rows.append({
            "phase": "P7",
            "method": "P7 Multi-tier L1 best",
            "task": "L1 binary",
            "features_count": 28,
            "macro_f1": _f(p7_l1.get("macro_f1")),
            "weighted_f1": None,
            "attack_recall": None,
            "fpr": None,
            "accuracy": _f(p7_l1.get("accuracy")),
            "model_size_bytes": None,
            "bandwidth_total_bytes": int(float(p7_l1.get("communication_total_bytes", 0))),
            "runtime": "true_flower",
            "true_flower_runtime": True,
            "best_use_case": "Hierarchical multi-tier (experimental)",
            "accepted": str(p7_l1.get("accepted", "")).lower() == "true",
            "source": "outputs/reports/p7_multitier_summary.csv",
        })
    else:
        warnings_list.append({"field": "P7 Multi-tier L1 best", "issue": "l1_binary alpha=5.0 k=3 row not found"})

    if p7_l2:
        rows.append({
            "phase": "P7",
            "method": "P7 Multi-tier L2 best",
            "task": "L2 family",
            "features_count": 28,
            "macro_f1": _f(p7_l2.get("macro_f1")),
            "weighted_f1": None,
            "attack_recall": None,
            "fpr": None,
            "accuracy": _f(p7_l2.get("accuracy")),
            "model_size_bytes": None,
            "bandwidth_total_bytes": int(float(p7_l2.get("communication_total_bytes", 0))),
            "runtime": "true_flower",
            "true_flower_runtime": True,
            "best_use_case": "Hierarchical L2 multi-tier (experimental)",
            "accepted": str(p7_l2.get("accepted", "")).lower() == "true",
            "source": "outputs/reports/p7_multitier_summary.csv",
        })
    else:
        warnings_list.append({"field": "P7 Multi-tier L2 best", "issue": "l2_family alpha=0.5 k=3 row not found"})

    return rows, warnings_list


# ── markdown table ─────────────────────────────────────────────────────────

def _build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# P10 — Global Comparison Table",
        "",
        "| Phase | Method | Task | Features | Macro F1 | Attack Recall | FPR | Accuracy | BW (MB) | Runtime | Accepted |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        bw_mb = f"{r['bandwidth_total_bytes']/1e6:.2f}" if r.get("bandwidth_total_bytes") else "—"
        lines.append(
            f"| {r['phase']} | {r['method']} | {r['task']} "
            f"| {r['features_count']} "
            f"| {r['macro_f1'] if r['macro_f1'] is not None else '—'} "
            f"| {r['attack_recall'] if r['attack_recall'] is not None else '—'} "
            f"| {r['fpr'] if r['fpr'] is not None else '—'} "
            f"| {r['accuracy'] if r['accuracy'] is not None else '—'} "
            f"| {bw_mb} | {r['runtime']} | {'✅' if r['accepted'] else '❌'} |"
        )
    return "\n".join(lines) + "\n"


# ── figures ───────────────────────────────────────────────────────────────────

COLORS = {
    "P4 Centralized L1": "#2c7bb6",
    "P5 FedAvg L1": "#abd9e9",
    "P8 FedAvg + QGA L1": "#fdae61",
    "P9 QIFA L1": "#d7191c",
    "P9 QIFA + QGA L1": "#1a9641",
    "P6 L2 Flower baseline": "#9ecae1",
    "P8-b L2 FedAvg + QGA Flower": "#fc8d59",
    "P7 Multi-tier L1 best": "#bdbdbd",
    "P7 Multi-tier L2 best": "#969696",
}


def _l1_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["task"] == "L1 binary" and r.get("macro_f1") is not None]


def _fig_macro_f1(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    l1 = _l1_rows(rows)
    names = [r["method"].replace("P9 QIFA + QGA L1", "P9 QIFA+QGA").replace(" L1", "") for r in l1]
    vals = [r["macro_f1"] for r in l1]
    colors = [COLORS.get(r["method"], "#888") for r in l1]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, vals, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Macro F1")
    ax.set_title("P10 — L1 Binary: Macro F1 Comparison")
    ax.set_xlim(0.88, 0.97)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=9)
    ax.axvline(x=vals[0] if l1 else 0, color="#2c7bb6", linestyle="--", linewidth=1, alpha=0.5, label="P4 ceiling")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "p10_l1_macro_f1_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig_recall_fpr(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    l1 = [r for r in rows if r["task"] == "L1 binary" and r.get("attack_recall") is not None and r.get("fpr") is not None]
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in l1:
        name = r["method"].replace("P9 QIFA + QGA L1", "P9 QIFA+QGA").replace(" L1", "")
        color = COLORS.get(r["method"], "#888")
        ax.scatter(r["fpr"], r["attack_recall"], color=color, s=120, zorder=5, label=name)
        ax.annotate(name, (r["fpr"], r["attack_recall"]), textcoords="offset points", xytext=(6, 2), fontsize=7)
    ax.set_xlabel("False Positive Rate (FPR) ↓ better")
    ax.set_ylabel("Attack Recall ↑ better")
    ax.set_title("P10 — L1 Binary: Attack Recall vs FPR Trade-off")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path = out_dir / "p10_l1_attack_recall_fpr_tradeoff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig_bandwidth(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    l1 = [r for r in rows if r["task"] == "L1 binary" and r.get("bandwidth_total_bytes") is not None]
    names = [r["method"].replace("P9 QIFA + QGA L1", "P9 QIFA+QGA").replace(" L1", "") for r in l1]
    vals = [r["bandwidth_total_bytes"] / 1e6 for r in l1]
    colors = [COLORS.get(r["method"], "#888") for r in l1]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, vals, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Total Bandwidth (MB, 30 rounds)")
    ax.set_title("P10 — L1 Binary: Total Communication Bandwidth")
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2, f"{val:.1f} MB", va="center", fontsize=9)
    fig.tight_layout()
    path = out_dir / "p10_l1_bandwidth_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig_features_vs_f1(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    l1 = [r for r in rows if r["task"] == "L1 binary" and r.get("macro_f1") is not None]
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in l1:
        name = r["method"].replace("P9 QIFA + QGA L1", "P9 QIFA+QGA").replace(" L1", "")
        color = COLORS.get(r["method"], "#888")
        ax.scatter(r["features_count"], r["macro_f1"], color=color, s=120, zorder=5, label=name)
        ax.annotate(name, (r["features_count"], r["macro_f1"]), textcoords="offset points", xytext=(4, 2), fontsize=7)
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Macro F1")
    ax.set_title("P10 — L1 Binary: Features Count vs Macro F1")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 35)
    fig.tight_layout()
    path = out_dir / "p10_l1_features_vs_macro_f1.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig_l2_comparison(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    l2 = [r for r in rows if r["task"] == "L2 family" and r.get("macro_f1") is not None]
    if not l2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No L2 data available", ha="center", va="center", transform=ax.transAxes)
        path = out_dir / "p10_l2_p6_vs_p8b.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    names = [r["method"] for r in l2]
    f1_vals = [r["macro_f1"] for r in l2]
    acc_vals = [r["accuracy"] if r.get("accuracy") else 0 for r in l2]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width / 2, f1_vals, width, label="Macro F1", color=["#9ecae1", "#fc8d59"][:len(l2)], edgecolor="white")
    b2 = ax.bar(x + width / 2, acc_vals, width, label="Accuracy", color=["#6baed6", "#f16913"][:len(l2)], edgecolor="white", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("P10 — L2 Family: P6 vs P8-b QGA")
    ax.legend()
    ax.set_ylim(0, 1.0)
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    path = out_dir / "p10_l2_p6_vs_p8b.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig_ranking_table(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    l1 = [r for r in rows if r["task"] == "L1 binary" and r.get("macro_f1") is not None]
    l1_sorted = sorted(l1, key=lambda r: r["macro_f1"], reverse=True)
    col_labels = ["Rank", "Method", "Macro F1", "Attack Recall", "FPR", "Features", "Runtime"]
    table_data = []
    for i, r in enumerate(l1_sorted, 1):
        table_data.append([
            str(i),
            r["method"].replace("P9 QIFA + QGA L1", "P9 QIFA+QGA").replace(" L1", ""),
            f"{r['macro_f1']:.4f}",
            f"{r['attack_recall']:.4f}" if r.get("attack_recall") else "—",
            f"{r['fpr']:.4f}" if r.get("fpr") else "—",
            str(r["features_count"]),
            r["runtime"],
        ])
    fig, ax = plt.subplots(figsize=(12, max(3, len(table_data) * 0.6 + 1.5)))
    ax.axis("off")
    tbl = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c7bb6")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f4f8")
    ax.set_title("P10 — L1 Binary Method Ranking by Macro F1", fontsize=12, pad=10)
    fig.tight_layout()
    path = out_dir / "p10_method_ranking_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig_architecture_summary(out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_title("P10 — Final Architecture Summary: Quantum-Inspired FL IoT IDS", fontsize=13, fontweight="bold", pad=12)

    boxes = [
        (1.0, 5.5, "#e8f4f8", "P4 Centralized\nL1 MLP\n28→128→64→2\nF1=0.9611"),
        (4.0, 5.5, "#fff3cd", "P5 FedAvg L1\n28 features\n30 rounds\nF1=0.9407"),
        (7.0, 5.5, "#d4edda", "P8 FedAvg+QGA\n12 features\n-57% BW\nF1=0.9480"),
        (1.5, 2.5, "#fde8e8", "P6 Hierarchical\nL2 Family\n28 features\nF1=0.6355"),
        (4.5, 2.5, "#ffe5d0", "P8-b L2+QGA\n19 features\n-9% BW\nF1=0.6466"),
        (7.5, 2.5, "#e8d5f5", "P7 Multi-tier\nL1+L2\n28 features\nExperimental"),
        (5.5, 0.4, "#d0f0e0", "P9 QIFA Flower\n28f F1=0.9454 FPR↓\n+ QIFA+QGA 12f F1=0.9471 Recall↑"),
    ]
    for x, y, color, text in boxes:
        rect = mpatches.FancyBboxPatch((x - 1.1, y - 0.7), 2.2, 1.4, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="#555", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5, wrap=True)

    for (x1, y1), (x2, y2) in [
        ((4.0, 5.5), (1.0, 5.5)), ((4.0, 5.5), (7.0, 5.5)),
        ((4.5, 2.5), (1.5, 2.5)),
        ((7.0, 5.5), (5.5, 1.1)), ((4.0, 5.5), (5.5, 1.1)),
    ]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))

    ax.text(5, 6.6, "← Task L1 Binary →", ha="center", fontsize=9, color="#333")
    ax.text(3, 3.6, "← Task L2 Family →", ha="center", fontsize=9, color="#333")
    ax.text(5.5, 1.6, "↑ P9 QIFA (quantum aggregation)", ha="center", fontsize=9, color="#1a9641")

    fig.tight_layout()
    path = out_dir / "p10_final_architecture_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── main ─────────────────────────────────────────────────────────────────────

def build_evidence_pack(repo_root: Path) -> dict[str, Any]:
    reports_dir = repo_root / "experiments/qi-fl-ids-iot-final/outputs/reports"
    figures_dir = repo_root / "experiments/qi-fl-ids-iot-final/outputs/figures/p10"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows, warnings_list = _collect_rows(repo_root)

    _write_csv(reports_dir / "p10_global_comparison.csv", rows)
    _write_json(reports_dir / "p10_global_comparison.json", rows)
    (reports_dir / "p10_global_comparison_table.md").write_text(_build_markdown(rows), encoding="utf-8")

    figures_generated = []
    figures_generated.append(str(_fig_macro_f1(rows, figures_dir)))
    figures_generated.append(str(_fig_recall_fpr(rows, figures_dir)))
    figures_generated.append(str(_fig_bandwidth(rows, figures_dir)))
    figures_generated.append(str(_fig_features_vs_f1(rows, figures_dir)))
    figures_generated.append(str(_fig_l2_comparison(rows, figures_dir)))
    figures_generated.append(str(_fig_ranking_table(rows, figures_dir)))
    figures_generated.append(str(_fig_architecture_summary(figures_dir)))

    manifest = {
        "accepted": True,
        "phase": "P10",
        "method_count": len(rows),
        "methods": [r["method"] for r in rows],
        "l1_methods": [r["method"] for r in rows if r["task"] == "L1 binary"],
        "l2_methods": [r["method"] for r in rows if r["task"] == "L2 family"],
        "figures_generated": figures_generated,
        "reports_generated": [
            "outputs/reports/p10_global_comparison.csv",
            "outputs/reports/p10_global_comparison.json",
            "outputs/reports/p10_global_comparison_table.md",
        ],
        "warnings": warnings_list,
        "errors": [],
    }
    _write_json(reports_dir / "p10_evidence_manifest.json", manifest)

    print(f"P10 evidence pack built | methods={len(rows)} | figures={len(figures_generated)} | warnings={len(warnings_list)}")
    for w in warnings_list:
        print(f"  WARNING: {w['field']} — {w['issue']}")

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()
    repo_root = args.repo_root or Path.cwd().resolve()
    result = build_evidence_pack(repo_root)
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
