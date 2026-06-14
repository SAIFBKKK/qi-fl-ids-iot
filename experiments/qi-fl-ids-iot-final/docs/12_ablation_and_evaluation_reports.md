# P12 - Ablation and Evaluation Reports

## Objective

P12 consolidates the final evidence across L1, L2, robustness, and compression. It distinguishes measured metrics from dry-run or estimated structural values.

## Methods Compared

- P4 Centralized L1
- P5 FedAvg L1
- P8 FedAvg + QGA L1
- P9 QIFA L1
- P9 QIFA + QGA L1
- P6 L2 Flower baseline
- P8-b L2 FedAvg + QGA Flower
- P7 HeteroFL L1/L2 best rows
- P10 poisoned robustness rows
- P11 FedTN/MPS rank dry-run rows

## Results L1

P8 FedAvg + QGA is the best production L1 compromise because it preserves strong Macro-F1 while reducing features and bandwidth. P9 QIFA + QGA gives the strongest attack recall.

## Results L2

L2 remains experimental. P8-b QGA L2 improves the L2 Flower baseline in Macro-F1 and model size, but it is not a dashboard deployment target.

## Robustness

P10 shows QIFA + QGA is the strongest method under label flipping, with the best robustness score and attack recall.

## Compression

P11 FedTN/MPS rank 8 is a structural dry-run. It reduces model size and estimated bandwidth, but no measured Macro-F1/Recall/FPR is attached without a checkpoint evaluation.

## Final Ranking by Objective

- Best production L1: P8 FedAvg + QGA.
- Best attack recall: P9 QIFA + QGA.
- Best FPR: P9 QIFA.
- Best poisoning robustness: P10 QIFA + QGA.
- Best structural compression: P11 FedTN/MPS rank 8.
- L2 experimental: P8-b QGA L2.

## Recommendation for P13 Dashboard

Use L1 production artifacts and expose QGA/FedAvg metrics as the stable dashboard baseline. Keep QIFA, robustness, L2, and FedTN/MPS as research evidence panels or appendix material unless a later deployment phase hardens them.
