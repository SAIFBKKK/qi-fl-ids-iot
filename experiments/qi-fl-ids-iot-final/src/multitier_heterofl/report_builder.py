"""P7 documentation and verify outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from multitier_heterofl.config import rel, write_json


def write_doc(path: Path, *, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P7 - Multi-tier HeteroFL",
        "",
        "## 1. Objectif",
        "Adapter le FL L1/L2 aux noeuds IoT heterogenes avec HeteroFL shared-supernet et slicing.",
        "",
        "## 2. Donnees",
        "L1 utilise les partitions P3 materialisees. L2 utilise les partitions P3 `index_only`. L3 est exclu.",
        "",
        "## 3. Tiers",
        "- weak : `28 -> 64 -> output_dim`",
        "- medium : `28 -> 128 -> 64 -> output_dim`",
        "- powerful : `28 -> 256 -> 128 -> output_dim`",
        "- supernet : `28 -> 256 -> 128 -> output_dim`",
        "",
        "## 4. Aggregation HeteroFL",
        "Chaque client met a jour uniquement ses slices. Le serveur effectue une moyenne ponderee par `num_examples` sur les slices couvertes et conserve les valeurs globales ailleurs.",
        "",
        "## 5. Commandes",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/07_verify_multitier_heterofl_setup.py --config experiments/qi-fl-ids-iot-final/configs/multitier_heterofl.yaml`",
        "",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/07_run_multitier_heterofl.py --config experiments/qi-fl-ids-iot-final/configs/multitier_heterofl.yaml --task l1 --mode smoke --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000`",
        "",
        "`python experiments/qi-fl-ids-iot-final/src/scripts/07_run_multitier_heterofl.py --config experiments/qi-fl-ids-iot-final/configs/multitier_heterofl.yaml --task l2 --mode smoke --alpha 0.5 --clients 3 --rounds 1 --max-samples-per-client 1000`",
        "",
        "## 6. Etat verify",
        f"accepted=`{summary.get('accepted', False)}`",
        "",
        "## 7. Conclusion P7",
        "P7 est experimental et ne modifie pas dashboard, Docker ou modules QI.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_verify_outputs(*, repo_root: Path, config: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    reports_dir = repo_root / config["outputs"]["reports_dir"]
    docs_path = repo_root / config["final_experiment_dir"] / "docs" / "07_multitier_heterofl.md"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "multitier_heterofl_verify_summary.json"
    write_json(summary_path, summary)
    write_doc(docs_path, summary=summary)
    return [rel(summary_path, repo_root), rel(docs_path, repo_root)]
