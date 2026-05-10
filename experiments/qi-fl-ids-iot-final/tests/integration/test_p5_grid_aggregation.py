from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = (
    REPO_ROOT
    / "experiments"
    / "qi-fl-ids-iot-final"
    / "src"
    / "scripts"
    / "05_3_aggregate_fl_grid_results.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("p5_grid_aggregate", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p5_grid_aggregator_imports() -> None:
    module = _load_module()
    assert hasattr(module, "aggregate")


def test_alpha_and_k_regimes() -> None:
    module = _load_module()
    assert module.alpha_regime(0.1) == "extreme_noniid"
    assert module.alpha_regime(0.5) == "realistic_noniid"
    assert module.alpha_regime(5.0) == "quasi_iid"
    assert module.k_regime(3) == "low"
    assert module.k_regime(4) == "medium"
    assert module.k_regime(5) == "high_client_count"


def test_bandwidth_formula_constants() -> None:
    module = _load_module()
    assert 2 * 3 * module.MODEL_SIZE_BYTES == 290_352
    assert 2 * 4 * module.MODEL_SIZE_BYTES == 387_136
    assert 2 * 5 * module.MODEL_SIZE_BYTES == 483_920
    assert 30 * 2 * 3 * module.MODEL_SIZE_BYTES == 8_710_560
    assert 30 * 2 * 4 * module.MODEL_SIZE_BYTES == 11_614_080
    assert 30 * 2 * 5 * module.MODEL_SIZE_BYTES == 14_517_600


def test_rank_rows_orders_by_macro_f1_then_fpr() -> None:
    module = _load_module()
    rows = [
        {"alpha": 0.5, "clients": 3, "macro_f1": 0.90, "fpr": 0.05, "bandwidth_total_bytes": 100},
        {"alpha": 0.1, "clients": 3, "macro_f1": 0.92, "fpr": 0.10, "bandwidth_total_bytes": 100},
        {"alpha": 5.0, "clients": 3, "macro_f1": 0.92, "fpr": 0.03, "bandwidth_total_bytes": 100},
    ]
    ranked = module.rank_rows(rows)
    by_alpha = {(row["alpha"], row["clients"]): row["scenario_rank"] for row in ranked}
    assert by_alpha[(5.0, 3)] == 1
    assert by_alpha[(0.1, 3)] == 2
    assert by_alpha[(0.5, 3)] == 3


def test_missing_artifact_detection(tmp_path: Path) -> None:
    module = _load_module()
    missing = module.scenario_missing_files(tmp_path)
    assert "artifacts/run_summary.json" in missing
    assert "artifacts/metrics_rounds.csv" in missing


def test_scenario_row_rejects_non_full_run(tmp_path: Path) -> None:
    module = _load_module()
    base = tmp_path / "outputs" / "fl_l1_fedavg" / "alpha_0.1" / "k3" / "artifacts"
    base.mkdir(parents=True)
    for name in [
        "metrics_rounds.csv",
        "metrics_clients.csv",
        "bandwidth_rounds.csv",
    ]:
        (base / name).write_text("round\n1\n", encoding="utf-8")
    (base / "run_summary.json").write_text('{"mode":"smoke","rounds":1}', encoding="utf-8")
    (base / "metrics_test.json").write_text("{}", encoding="utf-8")
    (base / "comparison_with_p4.json").write_text("{}", encoding="utf-8")
    config = {"outputs": {"run_dir": "outputs/fl_l1_fedavg"}}
    row, status = module.scenario_row(config, tmp_path, 0.1, 3, expected_rounds=30)
    assert row is None
    assert status["status"] == "incomplete"
