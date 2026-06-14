"""Reports and documentation helpers for P5 FedAvg L1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .scenario_loader import rel, write_json


def build_verify_summary(
    *,
    repo_root: Path,
    config: dict[str, Any],
    checks: dict[str, bool],
    scenario_checks: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Build the code-ready verify summary."""

    return {
        "accepted": all(checks.values()) and not warnings,
        "mode": "verify",
        "strategy": "FedAvg",
        "dataset_level": "l1_binary",
        "checks": checks,
        "scenario_checks": scenario_checks,
        "global_test_holdout": {
            "path": config["inputs"]["global_test_npz"],
            "partitioned_by_p5": False,
            "usage": "final global evaluation only after validation threshold selection",
        },
        "p4_comparison_ready": checks.get("p4_metrics_exist", False),
        "sample_policy": {
            "smoke_may_use_max_samples_per_client": True,
            "full_mode_does_not_use_smoke_sampling": checks.get("full_mode_does_not_use_smoke_sampling", False),
            "full_uses_all_client_samples": checks.get("full_uses_all_client_samples", False),
        },
        "runtime_logging": {
            "verbose_rounds": config.get("logging", {}).get("verbose_rounds", True),
            "verbose_clients": config.get("logging", {}).get("verbose_clients", False),
            "flower_like_logs": config.get("logging", {}).get("flower_like_logs", True),
            "run_console_log": "outputs/fl_l1_fedavg/alpha_{alpha}/k{k}/logs/run_console.log",
        },
        "full_training_launched": False,
        "grid_launched": False,
        "warnings": warnings,
    }


def write_reuse_notes(path: Path) -> None:
    """Write the controlled v3 reuse notes requested by P5."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# P5 Reuse Notes — v3 to final FedAvg L1",
                "",
                "## Fichiers v3 inspectés",
                "- `experiments/fl-iot-ids-v3/src/fl/reporting_strategy.py`",
                "- `experiments/fl-iot-ids-v3/src/fl/server_app.py`",
                "- `experiments/fl-iot-ids-v3/src/fl/client_app.py`",
                "- `experiments/fl-iot-ids-v3/src/fl/strategy.py`",
                "- `experiments/fl-iot-ids-v3/src/fl/aggregation_hooks.py`",
                "- `experiments/fl-iot-ids-v3/src/fl/metrics.py`",
                "- `experiments/fl-iot-ids-v3/src/model/train.py`",
                "- `experiments/fl-iot-ids-v3/src/model/evaluate.py`",
                "- `experiments/fl-iot-ids-v3/configs/fl/fedavg_30rounds.yaml`",
                "",
                "## Réutilisé conceptuellement",
                "- Agrégation FedAvg pondérée par `num_examples`.",
                "- Séparation client/server.",
                "- Logs par round et par client.",
                "- Suivi des coûts de communication à partir de la taille du modèle.",
                "- Pattern de métriques agrégées et d'artefacts CSV/JSON.",
                "",
                "## Réimplémenté pour le dossier final",
                "- Code in-process sans dépendance runtime Flower obligatoire pour le mode verify/smoke.",
                "- Modèle L1 binaire `28 -> 128 -> 64 -> 2` aligné avec P4.",
                "- Chargement direct des partitions P3 finales `l1_binary/alpha_x/kY`.",
                "- Protection explicite du global test holdout.",
                "- Comparaison P4 vs P5 via les artefacts P4 finalisés.",
                "",
                "## Non repris",
                "- QIFA/QIFA-guard, SCAFFOLD, FedProx, multi-tier et node profiling v3.",
                "- Dépendances de chemins v3 et logique 34 classes.",
                "- Flower ServerApp/ClientApp runtime lourd, réservé à une éventuelle intégration ultérieure.",
                "",
                "## Pourquoi",
                "P5 doit être une baseline FedAvg L1 claire, reproductible et compatible avec les artefacts finaux P2/P3/P4, sans importer la complexité expérimentale v3 non nécessaire.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_code_ready_report(
    path: Path,
    *,
    summary: dict[str, Any],
    commands: dict[str, str],
) -> None:
    """Write docs/05_fl_baseline.md in code-ready state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P5 — Federated L1 FedAvg Baseline",
        "",
        "## 1. Objectif",
        "Créer la baseline Federated Learning L1 avec FedAvg pour comparaison avec P4 centralisé.",
        "",
        "## 2. Réutilisation contrôlée depuis v3",
        "Voir `outputs/reports/p5_reuse_notes.md`. Les patterns FedAvg, logging et communication tracking ont été repris conceptuellement, puis réimplémentés pour les artefacts finaux.",
        "",
        "## 3. Données utilisées",
        "P5 utilise les partitions L1 P3 pour train/val client. Le global test holdout P2 reste intact.",
        "",
        "## 4. Rappel P4 centralisé",
        "P4 sert de référence centralisée L1 et fournit `metrics_test.json`, `threshold.json` et `model_config.json`.",
        "",
        "## 5. Principe FedAvg",
        "`w_global = sum(n_k / n_total * w_k)` avec pondération par nombre d'exemples client.",
        "",
        "## 6. Scénarios Dirichlet",
        "La grille prévue est `alpha ∈ {0.1, 0.5, 5.0}` et `K ∈ {3, 4, 5}`.",
        "",
        "## 7. Configuration FL",
        "FedAvg, 30 rounds, local_epochs=1, client_fraction=1.0, batch_size=512.",
        "",
        "## 8. Architecture du modèle",
        "`28 -> 128 -> 64 -> 2`, alignée avec P4.",
        "",
        "## 9. Monitoring et communication cost",
        "Le code trace round metrics, client metrics, bandwidth, model size, aggregation time et latency.",
        "",
        "## 10. Métriques par round",
        "`metrics_rounds.csv` est généré après un run smoke/full/grid.",
        "",
        "## 11. Métriques par client",
        "`metrics_clients.csv` est généré après un run smoke/full/grid.",
        "",
        "## 12. Threshold tuning",
        "Le threshold est tuné sur validation fédérée uniquement.",
        "",
        "## 13. Évaluation global test holdout",
        "Le test global n'est évalué qu'après sélection du modèle et du threshold.",
        "",
        "## 14. Comparaison P4 vs P5",
        "`comparison_with_p4.json` est prêt après run et compare accuracy, macro-F1, attack recall et FPR.",
        "",
        "## 15. Impact de alpha",
        "À analyser après full/grid run utilisateur.",
        "",
        "## 16. Impact de K",
        "À analyser après full/grid run utilisateur.",
        "",
        "## 17. Artefacts générés",
        "- Code P5 prêt.",
        "- Verify summary : `outputs/reports/fl_l1_fedavg_verify_summary.json`.",
        "- Reuse notes : `outputs/reports/p5_reuse_notes.md`.",
        "",
        "## 18. Figures générées",
        "Les figures FL sont générées après smoke/full/grid run.",
        "",
        "## 19. Commandes d’exécution",
        f"- Verify : `{commands['verify']}`",
        f"- Smoke : `{commands['smoke']}`",
        f"- Full : `{commands['full']}`",
        f"- Grid : `{commands['grid']}`",
        "",
        "## 20. Critères d’acceptation",
        "",
        "| critere | ok |",
        "| --- | --- |",
    ]
    for key, value in summary["checks"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## 21. Conclusion P5",
            "",
            "P5 est code-ready. Aucun entraînement complet ni grid n'a été lancé par Codex; verify/smoke/full/grid sont disponibles pour exécution utilisateur.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_verify_outputs(
    *,
    repo_root: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> list[str]:
    """Write verify summary, reuse notes and docs."""

    reports_dir = repo_root / config["outputs"]["reports_dir"]
    docs_path = repo_root / config["final_experiment_dir"] / "docs" / "05_fl_baseline.md"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "fl_l1_fedavg_verify_summary.json"
    reuse_notes_path = reports_dir / "p5_reuse_notes.md"
    write_json(summary_path, summary)
    write_reuse_notes(reuse_notes_path)
    commands = {
        "verify": "python experiments/qi-fl-ids-iot-final/src/scripts/05_verify_fl_l1_setup.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_fedavg.yaml",
        "smoke": "python experiments/qi-fl-ids-iot-final/src/scripts/05_train_fl_l1_fedavg.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_fedavg.yaml --mode smoke --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000",
        "full": "python experiments/qi-fl-ids-iot-final/src/scripts/05_train_fl_l1_fedavg.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_fedavg.yaml --mode full --alpha 0.5 --clients 3 --rounds 30",
        "grid": "python experiments/qi-fl-ids-iot-final/src/scripts/05_train_fl_l1_fedavg.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_fedavg.yaml --mode grid",
    }
    write_code_ready_report(docs_path, summary=summary, commands=commands)
    return [rel(summary_path, repo_root), rel(reuse_notes_path, repo_root), rel(docs_path, repo_root)]
