from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_DIR = REPO_ROOT / "experiments" / "qi-fl-ids-iot-final"


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_p15_audit_plan_and_runtime_files_exist() -> None:
    assert (FINAL_DIR / "outputs" / "reports" / "p15_online_replay_audit.md").exists()
    assert (FINAL_DIR / "outputs" / "reports" / "p15_online_replay_plan.md").exists()

    assert (FINAL_DIR / "src" / "scripts" / "15_run_online_http_replay.py").exists()
    assert (FINAL_DIR / "src" / "scripts" / "15_check_mqtt_topics.py").exists()
    assert (FINAL_DIR / "src" / "scripts" / "15_collect_online_evidence.py").exists()

    validator_dir = FINAL_DIR / "deployment" / "online_validator"
    assert (validator_dir / "app.py").exists()
    assert (validator_dir / "mqtt_observer.py").exists()
    assert (validator_dir / "metrics.py").exists()
    assert (validator_dir / "Dockerfile").exists()
    assert (validator_dir / "requirements.txt").exists()

    compose_text = (FINAL_DIR / "deployment" / "docker-compose.final.yml").read_text(encoding="utf-8")
    assert "online-validator" in compose_text
    assert "8015:8015" in compose_text
    assert "profiles: [\"online\"]" in compose_text


def test_http_replay_dry_run_generates_reports(tmp_path: Path) -> None:
    run_command(
        [
            str(FINAL_DIR / "src" / "scripts" / "15_run_online_http_replay.py"),
            "--dry-run",
            "--max-rows",
            "2",
            "--use-qga-mask",
            "--reports-dir",
            str(tmp_path),
        ]
    )

    summary_path = tmp_path / "p15_online_http_replay_summary.json"
    predictions_path = tmp_path / "p15_online_http_replay_predictions.csv"
    table_path = tmp_path / "p15_online_http_replay_table.md"

    assert summary_path.exists()
    assert predictions_path.exists()
    assert table_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["use_qga_mask"] is True
    assert summary["selected_mask_id"] == "conservative_seed_42"
    assert summary["input_dim_sent"] == 12
    assert summary["rows_attempted"] == 2
    assert summary["test_used_for_training"] is False
    assert summary["test_used_for_model_selection"] is False


def test_evidence_collector_writes_reports_even_if_services_are_down(tmp_path: Path) -> None:
    run_command(
        [
            str(FINAL_DIR / "src" / "scripts" / "15_collect_online_evidence.py"),
            "--reports-dir",
            str(tmp_path),
            "--timeout-sec",
            "0.2",
        ]
    )

    evidence_path = tmp_path / "p15_online_evidence.json"
    table_path = tmp_path / "p15_online_evidence_table.md"
    assert evidence_path.exists()
    assert table_path.exists()

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["phase"] == "P15"
    assert "final_ids_api_health" in evidence["endpoints"]
    assert "dashboard_p13_health" in evidence["endpoints"]
