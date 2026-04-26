| Experiment | Strategy | Scenario | Imbalance | Rounds | Macro-F1 | Rare Macro-F1 | Benign Recall | FPR | Rare Recall | Accuracy | Recall Macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp_v3_fedavg_absentlocal_classweights | fedavg | absent_local | class_weights | 10 | 0.4709 | 0.2161 | 0.9278 | 0.0722 | 0.2642 | 0.5974 | 0.5241 |
| exp_v3_fedavg_normal_classweights | fedavg | normal_noniid | class_weights | 10 | 0.6676 | - | 0.9000 | 0.1000 | 0.4365 | 0.7165 | 0.7004 |
| exp_v3_fedavg_normal_focal | fedavg | normal_noniid | focal_loss | 10 | 0.6035 | - | 0.9161 | 0.0839 | 0.2335 | 0.6796 | 0.6405 |
| exp_v3_fedavg_normal_focal_weighted | fedavg | normal_noniid | focal_loss_weighted | 10 | 0.5661 | - | 0.9125 | 0.0875 | 0.3281 | 0.6415 | 0.6244 |
| exp_v3_fedavg_rareexpert_classweights | fedavg | rare_expert | class_weights | 10 | 0.2077 | 0.0587 | 0.9995 | 0.0005 | 0.0353 | 0.2864 | 0.2689 |
| exp_v3_fedprox_absentlocal_classweights | fedprox | absent_local | class_weights | 10 | 0.5238 | 0.2083 | 0.8826 | 0.1174 | 0.2086 | 0.5741 | 0.5729 |
| exp_v3_fedprox_normal_classweights | fedprox | normal_noniid | class_weights | 10 | 0.5875 | 0.2597 | 0.7835 | 0.2165 | 0.3036 | 0.6415 | 0.6236 |
| exp_v3_fedprox_rareexpert_classweights | fedprox | rare_expert | class_weights | 10 | 0.5179 | 0.3288 | 0.9861 | 0.0139 | 0.3143 | 0.5722 | 0.5520 |
| exp_v3_fedprox_rareexpert_focal | fedprox | rare_expert | focal_loss | 10 | 0.4980 | 0.2887 | 0.9808 | 0.0192 | 0.2721 | 0.5632 | 0.5338 |
| exp_v3_fedprox_rareexpert_focal_weighted | fedprox | rare_expert | focal_loss_weighted | 10 | 0.4670 | 0.3402 | 0.9589 | 0.0411 | 0.3484 | 0.5239 | 0.5099 |
| exp_v3_scaffold_absentlocal_classweights | scaffold | absent_local | class_weights | 10 | 0.0032 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0416 | 0.0294 |
| exp_v3_scaffold_normal_classweights | scaffold | normal_noniid | class_weights | 10 | 0.0022 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0114 | 0.0263 |
| exp_v3_scaffold_rareexpert_classweights | scaffold | rare_expert | class_weights | 10 | 0.0055 | 0.0234 | 0.0001 | 0.9999 | 0.0573 | 0.0130 | 0.0137 |
