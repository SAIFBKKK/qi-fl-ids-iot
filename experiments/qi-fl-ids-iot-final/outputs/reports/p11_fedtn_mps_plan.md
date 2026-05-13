# P11 FedTN/MPS Plan

## P11.1 Post-training compression

Build a dense 12 -> 128 -> 64 -> 2 L1 QGA model and replace target dense layers `fc1` and `fc2` with low-rank/MPS-style factorized layers.

## P11.2 Evaluation compressed model

If a local checkpoint exists, load it and evaluate validation/test metrics. If absent, produce structural estimates with clear warnings.

## P11.3 Optional fine-tuning

Fine-tuning is optional and should be launched manually. It is not part of the code-ready smoke.

## P11.4 Optional FL compressed runtime

Future FL runtime can transmit compressed weights or compressed checkpoints, but P11 code-ready does not launch Flower full automatically.

## P11.5 Comparison report

Compare dense and compressed models on:

- num_parameters
- model_size_bytes
- compression_ratio
- bandwidth_total_bytes
- estimated communication reduction
- Macro-F1 / attack recall / FPR when checkpoints and evaluation are available

## Constraints

- L1 only.
- No Docker or dashboard changes.
- No P12.
- No full FL run automatically.
- No checkpoints, logs, datasets, preprocessed arrays, partitions, or heavy runs committed.
