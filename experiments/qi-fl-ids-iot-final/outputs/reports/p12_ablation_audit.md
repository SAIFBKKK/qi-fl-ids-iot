# P12 Ablation Audit

## Files Found
- p8_qga_ablation_summary: FOUND C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\reports\p8_qga_ablation_summary.csv
- p8b_qga_l2_ablation_summary: FOUND C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\reports\p8b_qga_l2_ablation_summary.csv
- p9_qifa_ablation_summary: FOUND C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\reports\p9_qifa_ablation_summary.csv
- p10_robustness_full_summary: FOUND C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\reports\p10_robustness_full_summary.csv
- p10_robustness_clean_vs_poisoned: FOUND C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\reports\p10_robustness_clean_vs_poisoned.csv
- p11_fedtn_mps_summary: FOUND C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\reports\p11_fedtn_mps_summary.csv
- p5_grid_summary: FOUND C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\reports\p5_grid_summary.csv
- p7_multitier_summary: FOUND C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\reports\p7_multitier_summary.csv
- p10_global_comparison: FOUND C:\Users\saifb\dev\qi-fl-ids-iot\experiments\qi-fl-ids-iot-final\outputs\reports\p10_global_comparison.csv

## Missing Files
- None

## Measured vs Estimated
- P11 FedTN/MPS values are structural dry-run estimates, not measured Macro-F1/Recall/FPR.
- P10 robustness rows are measured full scenarios under label_flip poison_rate=0.2.
- P8/P9 L1 rows are measured full Flower/FedAvg evidence from existing reports.

## Metrics Availability
- p8_qga_ablation_summary: method, features_count, feature_reduction_ratio, macro_f1, attack_recall, fpr, accuracy, model_size_bytes, bandwidth_total_bytes, gap_macro_f1_vs_baseline, bandwidth_reduction_ratio, accepted, true_flower_runtime, runtime, mode, rounds, run_id, source_summary, notes
- p8b_qga_l2_ablation_summary: accepted, accuracy, bandwidth_reduction_ratio, bandwidth_total_bytes, feature_reduction_ratio, features_count, gap_macro_f1_vs_p6, macro_f1, macro_fpr, macro_precision, macro_recall, method, model_size_bytes, source, true_flower_runtime, weighted_f1
- p9_qifa_ablation_summary: accepted, accuracy, aggregation_type, alpha, attack_recall, bandwidth_total_bytes, clients, features_count, fpr, gamma, gap_macro_f1_vs_p5, gap_macro_f1_vs_p8, macro_f1, method, model_size_bytes, rounds, true_flower_runtime, variant, weighted_f1
- p10_robustness_full_summary: method, alpha, clients, attack_type, poison_rate, poisoned_clients, run_id, rounds, max_samples, run_type, scientific_use, macro_f1, attack_recall, fpr, fnr, accuracy, robustness_score, accepted
- p10_robustness_clean_vs_poisoned: method, clean_macro_f1, poisoned_macro_f1, delta_macro_f1, clean_attack_recall, poisoned_attack_recall, delta_attack_recall, clean_fpr, poisoned_fpr, delta_fpr, clean_accuracy, poisoned_accuracy, delta_accuracy, robustness_ratio_macro_f1, accepted
- p11_fedtn_mps_summary: base_model, rank, run_id, dry_run, dense_num_parameters, compressed_num_parameters, parameter_reduction_ratio, dense_model_size_bytes, compressed_model_size_bytes, compression_ratio, dense_bandwidth_total_bytes, compressed_bandwidth_total_bytes, bandwidth_reduction_ratio, macro_f1, attack_recall, fpr, accepted
- p5_grid_summary: alpha, clients, rounds, best_round, accuracy, macro_f1, attack_recall, fpr, fnr, weighted_f1, precision_attack, recall_attack, model_size_bytes, bandwidth_total_bytes, bandwidth_per_round_bytes, p4_accuracy, p4_macro_f1, p4_attack_recall, p4_fpr, gap_accuracy_vs_p4, gap_macro_f1_vs_p4, gap_attack_recall_vs_p4, gap_fpr_vs_p4, scenario_rank, alpha_regime, k_regime, scenario, output_dir
- p7_multitier_summary: task, alpha, clients, rounds, best_round, macro_f1, accuracy, communication_total_bytes, run_id, accepted
- p10_global_comparison: phase, method, task, features_count, macro_f1, weighted_f1, attack_recall, fpr, accuracy, model_size_bytes, bandwidth_total_bytes, runtime, true_flower_runtime, best_use_case, accepted, source
