# Baseline officielle

- Experiment: exp_flat_fedavg_rareexpert_focal
- Architecture: flat_34_v1style
- Strategy: fedavg
- Scenario: rare_expert
- Imbalance: focal_loss_weighted
- Num rounds: 10
- Completed rounds: 10
- Num clients: 3
- Feature count: 28
- Num classes: 34
- Duration sec: 978.84
- Status: success

## Fichiers
- run_summary.json
- round_metrics.json
- baseline_notes.md

## Final metrics
- final_distributed_loss: 34.24504945468928
- final_accuracy: 0.3969437533434663
- final_macro_f1: 0.2555616969167426
- final_recall_macro: 0.36087667684222774
- final_benign_recall: 0.242208859367206
- final_false_positive_rate: 0.757791140632794
- final_rare_class_recall: 0.14049739831985397
