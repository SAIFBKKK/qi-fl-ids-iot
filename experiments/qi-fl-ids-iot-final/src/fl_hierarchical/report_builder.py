"""Reports and documentation for P6 hierarchical Flower experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fl_l1.scenario_loader import rel
from fl_hierarchical.data import write_json


def write_hierarchical_doc(path: Path, *, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P6 - Hierarchical L2/L3 Flower FL Experiments",
        "",
        "## 1. Objectif",
        "P6 ajoute des experiences federated learning experimentales pour la classification L2 par famille d'attaque et L3 par type d'attaque.",
        "",
        "## 2. Positionnement",
        "Le modele L1 binaire reste le modele production et dashboard. L2 et L3 sont reserves a l'analyse scientifique et au rapport.",
        "",
        "## 3. Donnees utilisees",
        "P6 utilise `outputs/preprocessed/l2_family/` et les partitions P3 `outputs/partitions/l2_family/` en mode `index_only`.",
        "",
        "## 4. Runtime Flower",
        "Le runtime P6 reprend le mode Flower manuel/subprocess fiable de P5.2.1 : un vrai serveur Flower, de vrais clients Flower et des logs par `run_id`.",
        "",
        "## 5. P6.1 - L2 Family Flower FL",
        "L2 est une classification attack-only en 8 familles : BruteForce, DDoS, DoS, Malware, Mirai, Recon, Spoofing et Web-based.",
        "",
        "Architecture : `28 -> 128 -> 64 -> 8`.",
        "",
        "## 6. P6.2 - L3 Attack-Type Flower FL",
        "L3 derive `y_attack_type` depuis `label_id_original`, exclut BenignTraffic et remappe les 33 attaques vers `0..32`.",
        "",
        "Architecture : `28 -> 128 -> 64 -> 33`.",
        "",
        "## 7. Protection du global test holdout",
        "Le test global n'est jamais envoye aux clients. Il est charge uniquement cote serveur pour l'evaluation finale du modele global.",
        "",
        "## 8. Metriques",
        "Chaque run produit accuracy, macro-F1, weighted-F1, precision/recall macro, matrice de confusion et metriques one-vs-rest TP/FP/TN/FN par classe.",
        "",
        "## 9. Artefacts",
        "Chaque run ecrit `checkpoints/`, `artifacts/`, `logs/`, `run_summary.json`, `run_manifest.json`, `latest_run.json` et `latest_run_summary.json`.",
        "",
        "## 10. Figures",
        "Les figures sont ecrites sous `outputs/figures/hierarchical_flower/{task}/alpha_{alpha}/k{k}/{run_id}/`.",
        "",
        "## 11. Commandes verify et smoke",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/06_verify_hierarchical_flower_setup.py --config experiments/qi-fl-ids-iot-final/configs/hierarchical_flower.yaml`",
        "",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/06_run_hierarchical_flower_smoke.py --config experiments/qi-fl-ids-iot-final/configs/hierarchical_flower.yaml --task l2 --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000`",
        "",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/06_run_hierarchical_flower_smoke.py --config experiments/qi-fl-ids-iot-final/configs/hierarchical_flower.yaml --task l3 --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000`",
        "",
        "## 12. Commandes full manuelles",
        "Serveur L2 :",
        "",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/06_start_hierarchical_flower_server.py --config experiments/qi-fl-ids-iot-final/configs/hierarchical_flower.yaml --task l2 --alpha 0.5 --clients 3 --rounds 30 --address 127.0.0.1:8081`",
        "",
        "Clients L2 :",
        "",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/06_start_hierarchical_flower_client.py --config experiments/qi-fl-ids-iot-final/configs/hierarchical_flower.yaml --task l2 --alpha 0.5 --clients 3 --client-id client_1 --address 127.0.0.1:8081`",
        "",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/06_start_hierarchical_flower_client.py --config experiments/qi-fl-ids-iot-final/configs/hierarchical_flower.yaml --task l2 --alpha 0.5 --clients 3 --client-id client_2 --address 127.0.0.1:8081`",
        "",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/06_start_hierarchical_flower_client.py --config experiments/qi-fl-ids-iot-final/configs/hierarchical_flower.yaml --task l2 --alpha 0.5 --clients 3 --client-id client_3 --address 127.0.0.1:8081`",
        "",
        "Pour L3, utiliser `--task l3` et de preference un port different, par exemple `127.0.0.1:8082`.",
        "",
        "## 13. Baseline historique Kaggle L3",
        "La baseline Kaggle 34 classes historique (`28 -> 128 -> 64 -> 34`, test macro-F1 environ 0.811) reste une reference multiclasses, mais elle n'est pas le modele P6 Flower.",
        "",
        "## 14. Risques restants",
        "- Les NPZ L2 sont volumineux ; les full runs doivent etre lances manuellement.",
        "- Le mode legacy `start_server/start_client` est deprecie dans Flower, mais reste fonctionnel pour le runtime manuel local.",
        "- Les scores smoke ont une signification scientifique faible.",
        "",
        "## 15. Criteres d'acceptation",
        "P6 code-ready est accepte si verify, tests et smoke L2/L3 legers passent, avec `test_sent_to_clients=false` dans les summaries.",
        "",
        "## 16. Conclusion P6",
        f"Etat verify : accepted=`{summary.get('accepted', False)}`. P6 reste experimental et ne deploie pas L2/L3 dans le dashboard.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_verify_outputs(*, repo_root: Path, config: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    reports_dir = repo_root / config["outputs"]["reports_dir"]
    docs_path = repo_root / config["final_experiment_dir"] / "docs" / "06_hierarchical_l2_l3.md"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "hierarchical_flower_verify_summary.json"
    write_json(summary_path, summary)
    write_hierarchical_doc(docs_path, summary=summary)
    return [rel(summary_path, repo_root), rel(docs_path, repo_root)]
