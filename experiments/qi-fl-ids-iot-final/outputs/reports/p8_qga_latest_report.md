# P8 — QGA Feature Selection Report

## 1. Objective

Select a reduced L1 feature subset with a quantum-inspired genetic algorithm.

## 2. Run

- Run ID: `run_20260508_104311`
- Mode: `full`
- Accepted: `True`

## 3. Fitness

`0.6*macro_f1 + 0.3*attack_recall - 0.1*(features_count/28)`

## 4. Selected Features

Count: 9 / 28

- flow_duration
- Protocol Type
- Rate
- urg_count
- rst_count
- HTTPS
- ICMP
- IAT
- Number

## 5. Validation Metrics

- Macro-F1: 0.8948762078228165
- Attack recall: 0.9088686868686868
- FPR: 0.1198
- Fitness: 0.7774434736114388

## 6. Test Holdout

The global test holdout was not used for mask selection.

## 7. Artifacts

- `experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/runs/run_20260508_104311/artifacts/feature_mask.json`
- `experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/runs/run_20260508_104311/artifacts/feature_ranking.csv`
- `experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/runs/run_20260508_104311/artifacts/fitness_best.json`
- `experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/runs/run_20260508_104311/artifacts/fitness_weights.json`
- `experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/runs/run_20260508_104311/artifacts/qga_config.json`
- `experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/runs/run_20260508_104311/artifacts/qga_history.csv`
- `experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/runs/run_20260508_104311/artifacts/selected_features.json`
- `experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/runs/run_20260508_104311/artifacts/validation_metrics_best_mask.json`
- `experiments/qi-fl-ids-iot-final/outputs/qga_feature_selection/runs/run_20260508_104311/artifacts/run_summary.json`

## 8. Figures

- `experiments/qi-fl-ids-iot-final/outputs/figures/qga/run_20260508_104311/qga_fitness_evolution.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/qga/run_20260508_104311/qga_num_features_evolution.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/qga/run_20260508_104311/qga_selected_features_barplot.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/qga/run_20260508_104311/qga_feature_mask.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/qga/run_20260508_104311/qga_feature_importance_ranking.png`

## 9. Conclusion P8

P8 QGA standalone is code-ready when `accepted=true`; FedAvg/HeteroFL full validation remains a user-triggered step.
