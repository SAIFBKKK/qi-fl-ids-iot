# Reduced QI Benchmark Report Folder

This folder contains curated evidence for the reduced QI benchmark E1-E8 plus centralized reference R.

Current authoritative files:

- `run_status.csv`: final E1-E8 execution status. All E1-E8 rows are expected to be `success` with `completed_rounds=30`.
- `final_comparison_table.csv`: final comparison table used by the report.
- `final_report.md`: concise benchmark narrative and ablation interpretation.
- `figures/`: generated report figures.
- `confusion_matrices/`: confusion-matrix artifacts for the completed checkpoint evaluations currently covered by `evaluate_confusion_matrices.py`.
- `initial_run_inventory.csv`: historical pre-benchmark inventory. It is intentionally not the current final status.

Generated scratch files such as dry-run status CSVs, evaluator stdout/stderr logs, and exploratory per-class metric outputs should only be committed when explicitly promoted to final report evidence.
