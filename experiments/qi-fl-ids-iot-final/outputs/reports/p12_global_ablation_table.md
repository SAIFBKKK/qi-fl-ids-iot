| phase | method | task | macro_f1 | attack_recall | fpr | bandwidth | result_type |
|---|---|---|---:|---:|---:|---:|---|
| P4 | P4 Centralized L1 | l1_binary | 0.9611 | 0.9526 | 0.0293 | 0 | measured |
| P5 | P5 FedAvg L1 | l1_binary | 0.9407 | 0.9474 | 0.0663 | 8710560 | measured |
| P8 | P8 FedAvg + QGA L1 | l1_binary | 0.948 | 0.955 | 0.0594 | 7236000 | measured |
| P9 | P9 QIFA L1 | l1_binary | 0.9454 | 0.9436 | 0.0524 | 8710560 | measured |
| P9 | P9 QIFA + QGA L1 | l1_binary | 0.9471 | 0.9592 | 0.0658 | 7236000 | measured |
| P6 | P6 L2 Flower baseline | l2_family | 0.6355 | 0.7294 | 0.0439 | 8991360 | measured |
| P8-b | P8-b L2 FedAvg + QGA Flower | l2_family | 0.6466 | 0.7172 | 0.0332 | 8161920 | measured |
| P7 | P7 Multi-tier L1 best | l1_binary | 0.949 |  |  | 13118880 | measured |
| P7 | P7 Multi-tier L2 best | l2_family | 0.7085 |  |  | 13491840 | measured |
| P10 | P10 FedAvg poisoned | l1_binary_robustness | 0.9344680692677046 | 0.9185252525252525 | 0.04788888888888889 |  | measured |
| P10 | P10 FedAvg + QGA poisoned | l1_binary_robustness | 0.939086958704671 | 0.9368484848484848 | 0.058222222222222224 |  | measured |
| P10 | P10 QIFA poisoned | l1_binary_robustness | 0.9375599645294771 | 0.9425454545454546 | 0.0676 |  | measured |
| P10 | P10 QIFA + QGA poisoned | l1_binary_robustness | 0.9455302294225977 | 0.9565858585858585 | 0.06626666666666667 |  | measured |
| P11 | P11 FedTN/MPS fedavg_qga rank 8 | l1_binary_compression |  |  |  | 2144160 | dry_run |
