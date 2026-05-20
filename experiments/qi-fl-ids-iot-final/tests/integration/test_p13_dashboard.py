from __future__ import annotations

import csv
import json
from pathlib import Path


FINAL_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = FINAL_DIR / "outputs" / "reports"
DASHBOARD_DIR = FINAL_DIR / "dashboard"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_p13_dashboard_outputs_exist() -> None:
    assert (REPORTS_DIR / "p13_dashboard_audit.md").exists()
    assert (REPORTS_DIR / "p13_dashboard_plan.md").exists()
    assert (DASHBOARD_DIR / "model_registry.json").exists()
    assert (DASHBOARD_DIR / "data" / "dashboard_summary.json").exists()
    assert (REPORTS_DIR / "p13_dashboard_model_evaluation.csv").exists()
    assert (DASHBOARD_DIR / "app.py").exists() or (DASHBOARD_DIR / "templates" / "index.html").exists()


def test_p13_registry_and_recommendation() -> None:
    registry = json.loads((DASHBOARD_DIR / "model_registry.json").read_text(encoding="utf-8"))
    models = {model["model_id"]: model for model in registry["models"]}
    assert registry["recommended_model_id"] == "p8_fedavg_qga_l1"
    assert models["p8_fedavg_qga_l1"]["method"] == "FedAvg + QGA L1"
    assert models["p8_fedavg_qga_l1"]["recommended_use_case"] == "Recommended production L1"
    assert "p9_qifa_l1" in models
    assert "p9_qifa_qga_l1" in models


def test_p13_summary_contains_p10_and_p11_evidence() -> None:
    summary = json.loads((DASHBOARD_DIR / "data" / "dashboard_summary.json").read_text(encoding="utf-8"))
    assert summary["recommended_model"]["model_id"] == "p8_fedavg_qga_l1"
    assert summary["robustness"]["best_method"] == "qifa_qga"
    assert summary["robustness"]["best_row"]["method"] == "qifa_qga"
    assert summary["compression"]["result_type"] in {"dry_run", "structural"}
    assert summary["compression"]["row"].get("dry_run") in {"True", True, ""}


def test_p13_evaluations_include_required_models() -> None:
    rows = _read_csv(REPORTS_DIR / "p13_dashboard_model_evaluation.csv")
    model_ids = {row["model_id"] for row in rows}
    assert {"p8_fedavg_qga_l1", "p9_qifa_l1", "p9_qifa_qga_l1"}.issubset(model_ids)
    recommended = next(row for row in rows if row["model_id"] == "p8_fedavg_qga_l1")
    assert recommended["status"] in {"evaluable", "report_only"}
    assert recommended["macro_f1"] != ""


def test_p13_dashboard_is_local_no_docker_required() -> None:
    readme = (DASHBOARD_DIR / "README.md").read_text(encoding="utf-8")
    assert "python app.py" in readme
    assert "docker compose up" not in readme.lower()
