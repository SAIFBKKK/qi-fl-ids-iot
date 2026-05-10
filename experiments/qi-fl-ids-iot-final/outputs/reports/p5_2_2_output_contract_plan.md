# P5.2.2 - Flower Output Contract Plan

## 1. Architecture

Add two focused helpers:

- `summary_schema.py`: expected artifact/figure lists, verify contract enrichment, run criteria, rich run summary builder.
- `plotting.py`: deterministic matplotlib figures for Flower rounds, client metrics, thresholds, confusion matrix, ROC/PR, P4 comparison, and model architecture.

## 2. Verify Contract

`verify_flower_setup()` will write:

- `outputs/reports/fl_l1_flower_verify_summary.json`
- compatibility alias `outputs/reports/flower_l1_verify_summary.json`
- `docs/05_2_flower_runtime.md`

The verify JSON will contain `accepted`, `mode`, `flower_version`, `architecture`, `scenario`, `global_test_holdout`, `checks`, `artifacts_expected`, `figures_expected`, `criteria`, `warnings`, and `errors`.

## 3. Run Contract

Every smoke/full run writes:

- `artifacts/run_summary.json`
- `artifacts/run_manifest.json`
- scenario-level `latest_run_summary.json`

The run summary contains nested `scenario`, `dataset`, `model`, `training`, `threshold`, `validation`, `test`, `comparison_with_p4`, `artifacts`, `figures`, `criteria`, `warnings`, and `errors`.

## 4. Figures

Figures are written under:

`outputs/figures/fl_l1_flower/alpha_{alpha}/k{k}/{run_id}/`

The expected set contains round curves, bandwidth/runtime curves, evaluation plots, P4 comparison, client heatmap, and model architecture.

## 5. Compatibility

No P4/P5/P5.1 output is changed. Existing P5.2 run directories remain valid. New runs receive the richer contract.

## 6. Validation

Run only compile, light tests, verify, and one smoke round. Full 30-round Flower remains a user-run command.
