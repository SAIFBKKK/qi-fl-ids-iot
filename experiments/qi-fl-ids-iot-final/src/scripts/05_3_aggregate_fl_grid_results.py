"""Aggregate P5 FedAvg L1 alpha x K grid results.

This script is intentionally read-only with respect to training outputs. It
reads scenario artifacts produced by `05_train_fl_l1_fedavg.py` and writes
global comparative tables, figures, and the P5.3 analysis report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

import yaml  # noqa: E402


ALPHAS = [0.1, 0.5, 5.0]
CLIENTS = [3, 4, 5]
MODEL_SIZE_BYTES = 48_392
REQUIRED_ARTIFACTS = [
    "artifacts/run_summary.json",
    "artifacts/metrics_rounds.csv",
    "artifacts/metrics_clients.csv",
    "artifacts/bandwidth_rounds.csv",
    "artifacts/metrics_test.json",
    "artifacts/comparison_with_p4.json",
]


def alpha_dir(alpha: float) -> str:
    return f"alpha_{alpha:.1f}" if float(alpha).is_integer() else f"alpha_{alpha}"


def alpha_regime(alpha: float) -> str:
    if abs(alpha - 0.1) < 1e-9:
        return "extreme_noniid"
    if abs(alpha - 0.5) < 1e-9:
        return "realistic_noniid"
    if abs(alpha - 5.0) < 1e-9:
        return "quasi_iid"
    return "custom"


def k_regime(k: int) -> str:
    return {3: "low", 4: "medium", 5: "high_client_count"}.get(int(k), "custom")


def repo_rel(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def scenario_dir(config: dict[str, Any], repo_root: Path, alpha: float, clients: int) -> Path:
    return repo_root / config["outputs"]["run_dir"] / alpha_dir(alpha) / f"k{clients}"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    return config


def scenario_missing_files(base_dir: Path) -> list[str]:
    return [item for item in REQUIRED_ARTIFACTS if not (base_dir / item).exists()]


def scenario_row(
    config: dict[str, Any],
    repo_root: Path,
    alpha: float,
    clients: int,
    *,
    expected_rounds: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    base_dir = scenario_dir(config, repo_root, alpha, clients)
    missing = scenario_missing_files(base_dir)
    status = {
        "alpha": alpha,
        "K": clients,
        "status": "missing" if missing else "ready",
        "output_dir": repo_rel(base_dir, repo_root),
        "missing_files": missing,
    }
    if missing:
        return None, status

    run_summary = read_json(base_dir / "artifacts" / "run_summary.json")
    mode = str(run_summary.get("mode", ""))
    rounds = int(run_summary.get("rounds", 0))
    if mode not in {"full", "grid"} or rounds < int(expected_rounds):
        status["status"] = "incomplete"
        status["error_message"] = (
            f"Scenario exists but is not a full grid run: mode={mode!r}, "
            f"rounds={rounds}, expected_rounds={expected_rounds}"
        )
        return None, status
    metrics_test = read_json(base_dir / "artifacts" / "metrics_test.json")
    comparison = read_json(base_dir / "artifacts" / "comparison_with_p4.json")
    metrics_rounds = read_csv_rows(base_dir / "artifacts" / "metrics_rounds.csv")
    bandwidth_rounds = read_csv_rows(base_dir / "artifacts" / "bandwidth_rounds.csv")

    bandwidth_total = 0
    bandwidth_per_round = 2 * int(clients) * MODEL_SIZE_BYTES
    if bandwidth_rounds:
        last = bandwidth_rounds[-1]
        bandwidth_total = int(float(last.get("cumulative_bytes", 0)))
        bandwidth_per_round = int(float(last.get("total_bytes", bandwidth_per_round)))
    elif run_summary.get("round_rows"):
        last = run_summary["round_rows"][-1]
        bandwidth_total = int(last.get("communication_cumulative_bytes", 0))
        bandwidth_per_round = int(last.get("communication_total_bytes", bandwidth_per_round))

    row = {
        "alpha": alpha,
        "clients": clients,
        "rounds": int(run_summary.get("rounds", len(metrics_rounds))),
        "best_round": int(run_summary.get("best_round", 0)),
        "accuracy": float(metrics_test.get("accuracy", 0.0)),
        "macro_f1": float(metrics_test.get("macro_f1", 0.0)),
        "attack_recall": float(metrics_test.get("recall_attack", metrics_test.get("attack_recall", 0.0))),
        "fpr": float(metrics_test.get("FPR", metrics_test.get("fpr", 0.0))),
        "fnr": float(metrics_test.get("FNR", metrics_test.get("fnr", 0.0))),
        "weighted_f1": float(metrics_test.get("weighted_f1", 0.0)),
        "precision_attack": float(metrics_test.get("precision_attack", 0.0)),
        "recall_attack": float(metrics_test.get("recall_attack", metrics_test.get("attack_recall", 0.0))),
        "model_size_bytes": int(metrics_test.get("model_size_bytes", MODEL_SIZE_BYTES)),
        "bandwidth_total_bytes": bandwidth_total,
        "bandwidth_per_round_bytes": bandwidth_per_round,
        "p4_accuracy": float(comparison.get("p4_accuracy", 0.0)),
        "p4_macro_f1": float(comparison.get("p4_macro_f1", 0.0)),
        "p4_attack_recall": float(comparison.get("p4_attack_recall", 0.0)),
        "p4_fpr": float(comparison.get("p4_fpr", 0.0)),
        "gap_accuracy_vs_p4": float(comparison.get("gap_accuracy", 0.0)),
        "gap_macro_f1_vs_p4": float(comparison.get("gap_macro_f1", 0.0)),
        "gap_attack_recall_vs_p4": float(comparison.get("gap_attack_recall", 0.0)),
        "gap_fpr_vs_p4": float(comparison.get("gap_fpr", 0.0)),
        "scenario_rank": 0,
        "alpha_regime": alpha_regime(alpha),
        "k_regime": k_regime(clients),
        "scenario": f"alpha={alpha} K={clients}",
        "output_dir": repo_rel(base_dir, repo_root),
    }
    status["status"] = "ready"
    return row, status


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda item: (-float(item["macro_f1"]), float(item["fpr"]), int(item["bandwidth_total_bytes"])))
    for rank, row in enumerate(ranked, start=1):
        row["scenario_rank"] = rank
    return sorted(ranked, key=lambda item: (float(item["alpha"]), int(item["clients"])))


def make_markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "scenario_rank",
        "alpha",
        "clients",
        "alpha_regime",
        "macro_f1",
        "attack_recall",
        "fpr",
        "bandwidth_total_bytes",
        "gap_macro_f1_vs_p4",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in sorted(rows, key=lambda item: int(item["scenario_rank"])):
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def pivot(rows: list[dict[str, Any]], metric: str) -> list[list[float]]:
    by_key = {(float(row["alpha"]), int(row["clients"])): float(row[metric]) for row in rows}
    return [[by_key.get((alpha, k), float("nan")) for k in CLIENTS] for alpha in ALPHAS]


def heatmap(path: Path, rows: list[dict[str, Any]], metric: str, title: str, fmt: str = ".3f") -> Path:
    plt = _plt()
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(pivot(rows, metric), dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(CLIENTS)), labels=[f"K={k}" for k in CLIENTS])
    ax.set_yticks(range(len(ALPHAS)), labels=[f"alpha={a}" for a in ALPHAS])
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "NA" if np.isnan(value) else format(value, fmt)
            ax.text(j, i, label, ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def barplot_bandwidth(path: Path, rows: list[dict[str, Any]]) -> Path:
    plt = _plt()
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (float(row["alpha"]), int(row["clients"])))
    labels = [f"a={row['alpha']}\nK={row['clients']}" for row in ordered]
    values = [int(row["bandwidth_total_bytes"]) / (1024**2) for row in ordered]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#2563EB")
    ax.set_title("Total bandwidth by scenario")
    ax.set_ylabel("MB")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def line_by_round(path: Path, config: dict[str, Any], repo_root: Path, rows: list[dict[str, Any]], metric: str, title: str, ylabel: str) -> Path:
    plt = _plt()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for row in sorted(rows, key=lambda item: (float(item["alpha"]), int(item["clients"]))):
        rounds_path = scenario_dir(config, repo_root, float(row["alpha"]), int(row["clients"])) / "artifacts" / "metrics_rounds.csv"
        round_rows = read_csv_rows(rounds_path)
        if not round_rows:
            continue
        ax.plot(
            [int(item["round"]) for item in round_rows],
            [float(item[metric]) for item in round_rows],
            marker="o",
            linewidth=1.4,
            label=f"a={row['alpha']} K={row['clients']}",
        )
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def compare_p4(path: Path, rows: list[dict[str, Any]]) -> Path:
    plt = _plt()
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: int(row["scenario_rank"]))
    labels = [f"#{row['scenario_rank']}\na={row['alpha']} K={row['clients']}" for row in ordered]
    p5 = [float(row["macro_f1"]) for row in ordered]
    p4 = [float(ordered[0]["p4_macro_f1"]) for _ in ordered] if ordered else []
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(labels, p5, marker="o", label="P5 FedAvg macro-F1")
    ax.plot(labels, p4, linestyle="--", label="P4 centralized macro-F1")
    ax.set_title("P4 vs all P5 scenarios")
    ax.set_ylabel("Macro-F1")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def ranking_table_figure(path: Path, rows: list[dict[str, Any]]) -> Path:
    plt = _plt()
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: int(row["scenario_rank"]))[:9]
    table_rows = [
        [
            int(row["scenario_rank"]),
            row["alpha"],
            int(row["clients"]),
            f"{float(row['macro_f1']):.4f}",
            f"{float(row['attack_recall']):.4f}",
            f"{float(row['fpr']):.4f}",
        ]
        for row in ordered
    ]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(table_rows) + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["rank", "alpha", "K", "macro-F1", "attack recall", "FPR"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    ax.set_title("Scenario ranking")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def generate_figures(config: dict[str, Any], repo_root: Path, rows: list[dict[str, Any]], figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated = [
        heatmap(figures_dir / "heatmap_macro_f1_alpha_k.png", rows, "macro_f1", "Macro-F1 alpha x K"),
        heatmap(figures_dir / "heatmap_attack_recall_alpha_k.png", rows, "attack_recall", "Attack recall alpha x K"),
        heatmap(figures_dir / "heatmap_fpr_alpha_k.png", rows, "fpr", "FPR alpha x K"),
        barplot_bandwidth(figures_dir / "barplot_bandwidth_total_by_scenario.png", rows),
        line_by_round(figures_dir / "macro_f1_by_round_grouped_by_alpha.png", config, repo_root, rows, "macro_f1", "Macro-F1 by round grouped by alpha", "Macro-F1"),
        line_by_round(figures_dir / "fpr_by_round_grouped_by_alpha.png", config, repo_root, rows, "FPR", "FPR by round grouped by alpha", "FPR"),
        compare_p4(figures_dir / "p4_vs_all_p5_scenarios.png", rows),
        ranking_table_figure(figures_dir / "scenario_ranking_table.png", rows),
        heatmap(figures_dir / "heatmap_gap_macro_f1_vs_p4.png", rows, "gap_macro_f1_vs_p4", "Gap macro-F1 vs P4"),
        heatmap(figures_dir / "heatmap_bandwidth_alpha_k.png", rows, "bandwidth_total_bytes", "Bandwidth total alpha x K", fmt=".0f"),
        heatmap(figures_dir / "best_round_alpha_k.png", rows, "best_round", "Best round alpha x K", fmt=".0f"),
    ]
    return generated


def best_scenario(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=lambda row: int(row["scenario_rank"]))[0]


def write_analysis_report(path: Path, rows: list[dict[str, Any]], figures: list[Path], repo_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = best_scenario(rows)
    best_text = (
        f"Le meilleur scénario disponible est alpha={best['alpha']} K={best['clients']} "
        f"avec macro-F1={float(best['macro_f1']):.4f}."
        if best
        else "Aucun scénario complet n'est disponible pour une conclusion expérimentale."
    )
    figure_lines = "\n".join(f"- `{repo_rel(path, repo_root)}`" for path in figures)
    lines = [
        "# P5.3 — FedAvg L1 Grid Study",
        "",
        "## 1. Objectif",
        "Étudier l'influence de `alpha` et `K` sur les performances de FedAvg L1.",
        "",
        "## 2. Effet de alpha",
        "Quand alpha augmente, les clients deviennent plus similaires, le client drift diminue, FedAvg converge plus facilement, la Macro-F1 devrait augmenter ou devenir plus stable, et le FPR devrait diminuer.",
        "",
        "Hypothèse attendue : alpha=0.1 est le stress-test non-IID le plus difficile, alpha=0.5 est le scénario réaliste principal, et alpha=5.0 est la référence quasi-IID plus stable.",
        "",
        "## 3. Effet de K",
        "Quand K augmente, plus de clients participent, la communication augmente, les données sont plus fragmentées, et la variance entre clients peut augmenter. L'agrégation peut cependant bénéficier d'une diversité client plus grande.",
        "",
        "## 4. Communication / bandwidth",
        "Formule utilisée : `C_round = 2 × K × model_size` avec `model_size = 48,392 bytes`.",
        "",
        "- K=3 -> 290,352 bytes/round -> 8,710,560 bytes pour 30 rounds",
        "- K=4 -> 387,136 bytes/round -> 11,614,080 bytes pour 30 rounds",
        "- K=5 -> 483,920 bytes/round -> 14,517,600 bytes pour 30 rounds",
        "",
        "## 5. Tableau comparatif",
        "Tableau global : `experiments/qi-fl-ids-iot-final/outputs/reports/p5_grid_comparative_table.md`.",
        "",
        "## 6. Discussion",
        best_text,
        "",
        "Le scénario le moins coûteux en communication est toujours K=3. Le choix final doit équilibrer Macro-F1, attack recall, FPR et bandwidth total.",
        "",
        "Figures générées :",
        figure_lines,
        "",
        "## 7. Conclusion",
        "La recommandation finale doit être confirmée après exécution complète des 9 scénarios. Par défaut, alpha=0.5 K=3 reste le scénario principal raisonnable si ses performances restent proches du meilleur tout en gardant le coût de communication le plus faible.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate(config_path: Path, *, allow_missing: bool = False, expected_rounds: int = 30) -> dict[str, Any]:
    repo_root = Path.cwd().resolve()
    config = load_config(config_path)
    reports_dir = repo_root / config["outputs"]["reports_dir"]
    figures_dir = repo_root / "experiments/qi-fl-ids-iot-final/outputs/figures/fl_l1_grid"
    docs_path = repo_root / config["final_experiment_dir"] / "docs" / "05_3_fl_grid_analysis.md"

    rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        for clients in CLIENTS:
            row, status = scenario_row(config, repo_root, alpha, clients, expected_rounds=expected_rounds)
            statuses.append(status)
            if row is not None:
                rows.append(row)
    missing = [status for status in statuses if status["status"] != "ready"]
    if missing and not allow_missing:
        missing_label = ", ".join(f"alpha={item['alpha']} K={item['K']}" for item in missing)
        raise RuntimeError(f"Missing grid scenarios: {missing_label}. Use --allow-missing for partial aggregation.")
    rows = rank_rows(rows)

    summary_csv = reports_dir / "p5_grid_summary.csv"
    summary_json = reports_dir / "p5_grid_summary.json"
    table_md = reports_dir / "p5_grid_comparative_table.md"
    status_json = reports_dir / "p5_grid_status.json"
    figures_manifest = reports_dir / "p5_grid_figures_manifest.json"

    write_csv(summary_csv, rows)
    write_json(
        summary_json,
        {
            "complete": len(rows) == 9,
            "scenario_count": len(rows),
            "expected_scenario_count": 9,
            "rows": rows,
            "missing": missing,
        },
    )
    table_md.write_text(make_markdown_table(rows), encoding="utf-8")
    write_json(status_json, statuses)

    figures = generate_figures(config, repo_root, rows, figures_dir) if rows else []
    write_json(
        figures_manifest,
        {
            "figures": [repo_rel(path, repo_root) for path in figures],
            "complete_grid": len(rows) == 9,
            "scenario_count": len(rows),
        },
    )
    write_analysis_report(docs_path, rows, figures, repo_root)
    return {
        "complete": len(rows) == 9,
        "scenario_count": len(rows),
        "reports": {
            "summary_csv": repo_rel(summary_csv, repo_root),
            "summary_json": repo_rel(summary_json, repo_root),
            "comparative_table_md": repo_rel(table_md, repo_root),
            "status_json": repo_rel(status_json, repo_root),
            "figures_manifest": repo_rel(figures_manifest, repo_root),
            "analysis_report": repo_rel(docs_path, repo_root),
        },
        "figures": [repo_rel(path, repo_root) for path in figures],
        "missing": missing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate P5 FedAvg L1 grid results")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--rounds", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = aggregate(args.config, allow_missing=bool(args.allow_missing), expected_rounds=int(args.rounds))
    print("P5.3 grid aggregation complete")
    print(f"complete: {result['complete']}")
    print(f"scenario_count: {result['scenario_count']}/9")
    for key, value in result["reports"].items():
        print(f"{key}: {value}")
    if result["missing"]:
        print("missing scenarios:")
        for item in result["missing"]:
            detail = item.get("error_message") or item.get("missing_files")
            print(f"- alpha={item['alpha']} K={item['K']}: {detail}")
    return 0 if result["complete"] or args.allow_missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
