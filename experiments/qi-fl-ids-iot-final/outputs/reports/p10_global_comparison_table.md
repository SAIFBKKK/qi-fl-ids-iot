# P10 — Global Comparison Table

| Phase | Method | Task | Features | Macro F1 | Attack Recall | FPR | Accuracy | BW (MB) | Runtime | Accepted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P4 | P4 Centralized L1 | L1 binary | 28 | 0.9611 | 0.9526 | 0.0293 | 0.9612 | — | centralized | ✅ |
| P5 | P5 FedAvg L1 | L1 binary | 28 | 0.9407 | 0.9474 | 0.0663 | 0.9409 | 8.71 | in_process | ✅ |
| P8 | P8 FedAvg + QGA L1 | L1 binary | 12 | 0.948 | 0.955 | 0.0594 | 0.9481 | 7.24 | manual | ✅ |
| P9 | P9 QIFA L1 | L1 binary | 28 | 0.9454 | 0.9436 | 0.0524 | 0.9455 | 8.71 | true_flower | ✅ |
| P9 | P9 QIFA + QGA L1 | L1 binary | 12 | 0.9471 | 0.9592 | 0.0658 | 0.9473 | 7.24 | true_flower | ✅ |
| P6 | P6 L2 Flower baseline | L2 family | 28 | 0.6355 | 0.7294 | 0.0439 | 0.6841 | 8.99 | true_flower | ✅ |
| P8-b | P8-b L2 FedAvg + QGA Flower | L2 family | 19 | 0.6466 | 0.7172 | 0.0332 | 0.7555 | 8.16 | true_flower | ✅ |
| P7 | P7 Multi-tier L1 best | L1 binary | 28 | 0.949 | — | — | 0.949 | 13.12 | true_flower | ✅ |
| P7 | P7 Multi-tier L2 best | L2 family | 28 | 0.7085 | — | — | 0.8194 | 13.49 | true_flower | ✅ |
