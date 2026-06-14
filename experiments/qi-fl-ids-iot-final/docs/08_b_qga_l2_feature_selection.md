# P8-b — QGA Feature Selection for L2 Family Classification

P8-b is experimental and applies QGA feature selection to the L2 attack-family task only.

The L2 fitness is `0.60*MacroF1 + 0.25*MacroRecall - 0.10*MacroFPR - 0.05*FeatureRatio`.

QGA mask selection uses train/validation only. The L2 global test holdout is reserved for final Flower evaluation.

L2 FedAvg + QGA final training must use a true Flower runtime with `test_sent_to_clients=false`.
