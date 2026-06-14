"""Reports for P5.2 Flower runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fl_l1.scenario_loader import rel, write_json


def write_flower_runtime_doc(path: Path, *, summary: dict[str, Any]) -> None:
    """Write docs/05_2_flower_runtime.md."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P5.2 - True Flower L1 FedAvg Runtime",
        "",
        "## 1. Objectif",
        "Ajouter une voie Flower réelle à côté de la simulation FedAvg in-process P5.",
        "",
        "## 2. Architecture choisie",
        "P5.2 fournit `ClientApp` + `ServerApp` avec `flwr.simulation.run_simulation`, compatible avec Flower 1.8.0 installé localement.",
        "",
        "Pour le smoke fiable sur ce poste Windows, P5.2 utilise aussi le fallback Flower legacy-local : vrai `flwr.server.start_server` + vrais clients `flwr.client.start_client` sur localhost.",
        "",
        "## 3. Données utilisées",
        "Les clients utilisent uniquement les partitions L1 P3 `train_scaled.npz` et `val_scaled.npz`.",
        "",
        "## 4. Protection du global test holdout",
        "Le global test holdout n'est jamais envoyé aux clients. Il est chargé côté serveur uniquement après sélection validation.",
        "",
        "## 5. FedAvg Flower",
        "La stratégie `FlowerL1FedAvgStrategy` hérite de `flwr.server.strategy.FedAvg` et conserve les artefacts P5.",
        "",
        "## 6. Metrics et bandwidth",
        "Les métriques round/client, coûts de communication et comparaisons P4 sont écrits dans `outputs/fl_l1_flower/` après smoke/full.",
        "",
        "## 7. Commandes",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/05_2_verify_flower_l1_setup.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_flower.yaml`",
        "",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/05_2_run_flower_l1_smoke.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_flower.yaml --runtime legacy-local --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000`",
        "",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/05_2_run_flower_l1_smoke.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_flower.yaml --runtime legacy-local --mode full --alpha 0.5 --clients 3 --rounds 30`",
        "",
        "## 8. P5.2.1 Manual Flower Runtime",
        "Le blocage observe en legacy-local full est probablement lie au lancement serveur/clients Flower dans le meme processus Windows, avec des threads Python et des logs scenario-level reutilises entre smoke et full.",
        "",
        "P5.2.1 garde la voie Flower existante, mais ajoute une execution manuelle fiable : un vrai serveur Flower dans un terminal et un vrai client Flower par terminal. Les logs sont maintenant isoles par `run_id` sous `outputs/fl_l1_flower/alpha_0.5/k3/runs/{run_id}/logs/`.",
        "",
        "Commande serveur :",
        "",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/05_2_start_flower_server.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_flower.yaml --alpha 0.5 --clients 3 --rounds 30 --address 127.0.0.1:8080`",
        "",
        "Commandes clients :",
        "",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/05_2_start_flower_client.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_flower.yaml --alpha 0.5 --clients 3 --client-id client_1 --address 127.0.0.1:8080`",
        "",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/05_2_start_flower_client.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_flower.yaml --alpha 0.5 --clients 3 --client-id client_2 --address 127.0.0.1:8080`",
        "",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/05_2_start_flower_client.py --config experiments/qi-fl-ids-iot-final/configs/fl_l1_flower.yaml --alpha 0.5 --clients 3 --client-id client_3 --address 127.0.0.1:8080`",
        "",
        "Le script PowerShell optionnel `experiments/qi-fl-ids-iot-final/scripts/run_flower_l1_manual.ps1` ouvre les quatre terminaux, mais il n'est pas lance automatiquement.",
        "",
        "Pour verifier le port sous Windows : `netstat -ano | findstr :8080`. Pour arreter une execution bloquee, fermer les terminaux Flower ou tuer les PID identifies par `netstat`.",
        "",
        "Les clients ne chargent jamais `test_scaled.npz`. Le global test holdout reste cote serveur pour l'evaluation finale et le manifest ecrit `test_sent_to_clients=false`.",
        "",
        "Limite runtime : Flower 1.8.0 affiche des warnings deprecation pour `start_server` et `start_client`, mais ces API restent fonctionnelles pour le mode manuel demonstratif.",
        "",
        "## 9. P5.2.2 Output Contract and Reporting",
        "P5.2.2 separe deux niveaux de sortie : le verify summary et le run summary.",
        "",
        "Le verify summary est un contrat readiness leger. Il verifie Flower, les partitions P3, les metriques P4, la protection du global test holdout, puis liste `artifacts_expected`, `figures_expected`, `criteria`, `warnings` et `errors` dans `outputs/reports/fl_l1_flower_verify_summary.json`.",
        "",
        "Le run summary est le contrat scientifique par execution. Chaque smoke/full ecrit `runs/{run_id}/artifacts/run_summary.json` et une copie scenario-level `latest_run_summary.json`. Il contient les sections `scenario`, `dataset`, `model`, `training`, `threshold`, `validation`, `test`, `comparison_with_p4`, `artifacts`, `figures`, `criteria`, `warnings` et `errors`.",
        "",
        "Les artefacts de run attendus sont dans `checkpoints/`, `artifacts/` et `logs/` sous `outputs/fl_l1_flower/alpha_{alpha}/k{k}/runs/{run_id}/`. Les figures sont generees sous `outputs/figures/fl_l1_flower/alpha_{alpha}/k{k}/{run_id}/`.",
        "",
        "Pour un smoke, le summary marque `scientific_significance=low_for_smoke`. Les fichiers sont utiles pour tester le runtime et le reporting, mais les scores ne doivent pas etre interpretes comme une baseline finale.",
        "",
        "## 10. État code-ready",
        f"- accepted: `{summary.get('accepted', False)}`",
        f"- flower_version: `{summary.get('flower_version', 'unknown')}`",
        f"- architecture: `{summary.get('architecture', 'ClientApp/ServerApp')}`",
        "",
        "## 11. Conclusion P5.2",
        "P5.2 est code-ready lorsque verify, tests et smoke léger passent. Aucun full run n'est lancé automatiquement.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_verify_outputs(*, repo_root: Path, config: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    """Write verify summary and P5.2 documentation."""

    reports_dir = repo_root / config["outputs"]["reports_dir"]
    docs_path = repo_root / config["final_experiment_dir"] / "docs" / "05_2_flower_runtime.md"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "fl_l1_flower_verify_summary.json"
    compatibility_summary_path = reports_dir / "flower_l1_verify_summary.json"
    write_json(summary_path, summary)
    write_json(compatibility_summary_path, summary)
    write_flower_runtime_doc(docs_path, summary=summary)
    return [rel(summary_path, repo_root), rel(compatibility_summary_path, repo_root), rel(docs_path, repo_root)]
