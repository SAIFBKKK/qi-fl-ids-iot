from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.common.paths import OUTPUTS_DIR, ROOT_DIR


REPORT_DIR = OUTPUTS_DIR / "reports" / "qi_benchmark_reduced"
FIGURE_DIR = REPORT_DIR / "figures"
BASELINE_DIR = OUTPUTS_DIR / "reports" / "baselines"
MODEL_FACTORY_DIR = OUTPUTS_DIR / "model_factory_30rounds"


@dataclass(frozen=True)
class BenchExperiment:
    experiment_id: str
    name: str
    scenario: str
    aggregation: str
    features: str


EXPERIMENTS = [
    BenchExperiment("E1", "exp_bench30_normal_fedavg_28f", "normal_noniid", "FedAvg", "28"),
    BenchExperiment("E2", "exp_bench30_normal_qifa_28f", "normal_noniid", "QIFA", "28"),
    BenchExperiment("E3", "exp_bench30_normal_fedavg_qga15", "normal_noniid", "FedAvg", "QGA-15"),
    BenchExperiment("E4", "exp_bench30_normal_qifa_qga15", "normal_noniid", "QIFA", "QGA-15"),
    BenchExperiment("E5", "exp_bench30_absent_fedavg_28f", "absent_local", "FedAvg", "28"),
    BenchExperiment("E6", "exp_bench30_absent_qifa_28f", "absent_local", "QIFA", "28"),
    BenchExperiment("E7", "exp_bench30_absent_fedavg_qga15", "absent_local", "FedAvg", "QGA-15"),
    BenchExperiment("E8", "exp_bench30_absent_qifa_qga15", "absent_local", "QIFA", "QGA-15"),
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _round_rows(exp: BenchExperiment) -> list[dict[str, Any]]:
    payload = _load_json(BASELINE_DIR / exp.name / "round_metrics.json")
    if payload is None:
        return []
    return list(payload.get("rounds", []))


def _summary(exp: BenchExperiment) -> dict[str, Any] | None:
    return _load_json(BASELINE_DIR / exp.name / "run_summary.json")


def _last_round(exp: BenchExperiment) -> dict[str, Any]:
    rows = _round_rows(exp)
    return rows[-1] if rows else {}


def _centralized_reference() -> dict[str, Any]:
    metrics = _load_json(MODEL_FACTORY_DIR / "powerful" / "metrics.json") or {}
    validation = dict(metrics.get("validation", {}))
    return {
        "experiment_id": "R",
        "experiment_name": "model_factory_30rounds_powerful",
        "scenario": "normal_noniid",
        "aggregation": "Centralized",
        "features": "28",
        "rounds": 30,
        "macro_f1_final": validation.get("macro_f1"),
        "benign_recall_final": validation.get("benign_recall"),
        "false_positive_rate_final": validation.get("false_positive_rate"),
        "rare_class_recall_final": "",
        "rare_macro_f1_final": "",
        "accuracy_final": validation.get("accuracy"),
        "loss_final": "",
        "update_size_bytes_final": "",
        "status": "success" if validation else "missing",
    }


def _row_for_experiment(exp: BenchExperiment) -> dict[str, Any]:
    summary = _summary(exp)
    last = _last_round(exp)
    rounds = int(summary.get("completed_rounds", 0)) if summary else 0
    status = str(summary.get("status", "missing")) if summary else "missing"
    reusable = status == "success" and rounds >= 30
    return {
        "experiment_id": exp.experiment_id,
        "experiment_name": exp.name,
        "scenario": exp.scenario,
        "aggregation": exp.aggregation,
        "features": exp.features,
        "rounds": rounds,
        "macro_f1_final": last.get("macro_f1", summary.get("final_macro_f1") if summary else ""),
        "benign_recall_final": last.get("benign_recall", summary.get("final_benign_recall") if summary else ""),
        "false_positive_rate_final": last.get("false_positive_rate", summary.get("final_false_positive_rate") if summary else ""),
        "rare_class_recall_final": last.get("rare_class_recall", summary.get("final_rare_class_recall") if summary else ""),
        "rare_macro_f1_final": last.get("rare_macro_f1", summary.get("final_rare_macro_f1") if summary else ""),
        "accuracy_final": last.get("accuracy", summary.get("final_accuracy") if summary else ""),
        "loss_final": last.get("distributed_loss", summary.get("final_distributed_loss") if summary else ""),
        "update_size_bytes_final": last.get("update_size_bytes", ""),
        "status": "success_30_rounds" if reusable else status,
    }


def build_final_table() -> list[dict[str, Any]]:
    rows = [_row_for_experiment(exp) for exp in EXPERIMENTS]
    rows.append(_centralized_reference())
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "final_comparison_table.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _series(exp: BenchExperiment, metric: str) -> tuple[list[int], list[float]]:
    rounds = []
    values = []
    for row in _round_rows(exp):
        value = row.get(metric)
        if isinstance(value, (int, float)):
            rounds.append(int(row["round"]))
            values.append(float(value))
    return rounds, values


def _plot_convergence(metric: str, ylabel: str, output_name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ref = _centralized_reference()
    ref_value = ref["macro_f1_final"] if metric == "macro_f1" else None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    panels = [
        ("normal_noniid", [EXPERIMENTS[0], EXPERIMENTS[1]]),
        ("absent_local", [EXPERIMENTS[4], EXPERIMENTS[5]]),
    ]
    for ax, (scenario, experiments) in zip(axes, panels):
        plotted = False
        for exp in experiments:
            rounds, values = _series(exp, metric)
            if values:
                ax.plot(rounds, values, marker="o", label=f"{exp.aggregation} {exp.features}f")
                plotted = True
        if ref_value is not None:
            ax.axhline(float(ref_value), linestyle="--", color="black", label="Centralized R")
        if not plotted:
            ax.text(0.5, 0.5, "pending 30-round runs", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(scenario)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / output_name, dpi=180)
    plt.close(fig)


def plot_final_barplot(rows: list[dict[str, Any]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("macro_f1_final", "Macro-F1"),
        ("benign_recall_final", "Benign Recall"),
        ("false_positive_rate_final", "FPR"),
        ("rare_class_recall_final", "Rare Recall"),
    ]
    labels = [row["experiment_id"] for row in rows]
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, (key, label) in enumerate(metrics):
        vals = []
        for row in rows:
            value = row.get(key)
            vals.append(float(value) if isinstance(value, (int, float)) else np.nan)
        ax.bar(x + (idx - 1.5) * width, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Final metrics: E1-E8 + centralized R")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure3_final_barplot.png", dpi=180)
    plt.close(fig)


def plot_qifa_diversity() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for exp in (EXPERIMENTS[1], EXPERIMENTS[5]):
        rounds, values = _series(exp, "qifa/diversity_mean")
        if not values:
            rounds, values = _series(exp, "qifa_diversity_norm")
        if values:
            ax.plot(rounds, values, marker="o", label=f"{exp.scenario} {exp.aggregation}")
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "pending QIFA 30-round runs", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Round")
    ax.set_ylabel("QIFA diversity mean")
    ax.set_title("QIFA diversity across scenarios")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure5_qifa_diversity.png", dpi=180)
    plt.close(fig)


def plot_bandwidth_if_available() -> None:
    pairs = [(EXPERIMENTS[0], EXPERIMENTS[2]), (EXPERIMENTS[1], EXPERIMENTS[3]), (EXPERIMENTS[4], EXPERIMENTS[6]), (EXPERIMENTS[5], EXPERIMENTS[7])]
    available = all(_series(a, "update_size_bytes")[1] and _series(b, "update_size_bytes")[1] for a, b in pairs)
    if not available:
        return
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for full_exp, qga_exp in pairs:
        for exp in (full_exp, qga_exp):
            rounds, values = _series(exp, "update_size_bytes")
            ax.plot(rounds, values, marker="o", label=f"{exp.experiment_id} {exp.scenario} {exp.aggregation} {exp.features}")
    ax.set_xlabel("Round")
    ax.set_ylabel("update_size_bytes")
    ax.set_title("Communication cost: 28 features vs QGA-15")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure6_bandwidth.png", dpi=180)
    plt.close(fig)


def build_report(rows: list[dict[str, Any]]) -> None:
    completed = [row for row in rows if row["status"] in {"success_30_rounds", "success"}]
    lines = [
        "# Reduced QI Benchmark Final Report",
        "",
        "## 1. Objective",
        "Compare a focused 8-experiment FL matrix plus one centralized reference for the QI sprint.",
        "",
        "## 2. Experimental Design",
        "Two scenarios (`normal_noniid`, `absent_local`), two aggregations (FedAvg, QIFA), and two feature settings (28 features, QGA-15).",
        "",
        "## 3. Benchmark Matrix (E1-E8 + R)",
        "",
        "| ID | Scenario | Aggregation | Features | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['experiment_id']} | {row['scenario']} | {row['aggregation']} | {row['features']} | {row['status']} |")
    lines.extend([
        "",
        "## 4. Dataset Scenarios",
        "`normal_noniid` is the moderate non-IID scenario. `absent_local` removes local class coverage and should increase client diversity.",
        "",
        "## 5. Methods",
        "FedAvg is the classical baseline. QIFA applies normalized diversity-aware client weighting. QGA-15 uses a theta-vector quantum-inspired selector over the real 28-feature input.",
        "",
        "## 6. QIFA Formulation",
        "`epsilon_k = ||w_k - w_avg|| / (||w_avg|| + 1e-8)` and effective client weights are normalized before aggregation.",
        "",
        "## 7. QGA Feature-Selection Protocol",
        "The serious protocol is configured with `k_features=15`, `n_generations=20`, `pop_size=15`, `epochs=2`, `max_samples_per_class=2000`, `seed=42`.",
        "",
        "## 8. Results Tables",
        "See `final_comparison_table.csv`. Missing or incomplete runs are left pending.",
        "",
        "## 9. Figure 1 Analysis",
        "Macro-F1 convergence is plotted when 30-round round metrics exist.",
        "",
        "## 10. Figure 2 Analysis",
        "Loss convergence is plotted when distributed loss curves exist.",
        "",
        "## 11. Figure 3 Analysis",
        "Final barplot compares available final metrics for E1-E8 and R.",
        "",
        "## 12. Figure 4 Analysis",
        "Confusion matrices are generated by `evaluate_confusion_matrices.py` when best checkpoints exist.",
        "",
        "## 13. Figure 5 Analysis",
        "QIFA diversity curves compare E2 and E6 when both QIFA runs are available.",
        "",
        "## 14. Figure 6 Analysis",
        "Bandwidth figure is generated only if all QGA-15 comparison runs have valid communication metrics.",
        "",
        "## 15. Comparison With Centralized Reference R",
        "R comes from `outputs/model_factory_30rounds` and is centralized/model-factory validation, not an identical FL protocol. Interpret comparisons accordingly.",
        "",
        "## 16. Discussion",
        "The report intentionally avoids conclusions for pending runs. Once E1-E8 finish, compare QIFA vs FedAvg within each scenario and QGA-15 vs 28-feature communication and quality.",
        "",
        "## 17. Limitations",
        "Full benchmark conclusions require completed 30-round runs. Windows/Ray execution can be slow on CPU. Centralized R is a useful reference but not protocol-identical.",
        "",
        "## 18. Final Conclusion",
        f"Completed/reusable rows currently available: {len(completed)} / {len(rows)}. No fabricated results are reported.",
        "",
    ])
    (REPORT_DIR / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_final_table()
    _plot_convergence("macro_f1", "Macro-F1", "figure1_macro_f1_convergence.png")
    _plot_convergence("distributed_loss", "Distributed loss", "figure2_loss_convergence.png")
    plot_final_barplot(rows)
    plot_qifa_diversity()
    plot_bandwidth_if_available()
    build_report(rows)
    print(f"Reduced QI benchmark report -> {REPORT_DIR}")


if __name__ == "__main__":
    main()
