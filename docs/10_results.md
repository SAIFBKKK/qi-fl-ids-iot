# Results

The final selected practical model is **FedAvg + QGA**.

| Configuration | Features | Macro-F1 | Attack Recall | FPR | Bandwidth |
| ------------- | -------: | -------: | ------------: | --: | --------: |
| FedAvg | 28 | 0.9407 | 0.9474 | 0.0663 | 8,710,560 B |
| FedAvg + QGA | 12 | 0.9480 | 0.9550 | 0.0594 | 7,236,000 B |
| QIFA | 28 | 0.9454 | 0.9436 | 0.0524 | 8,710,560 B |
| QIFA + QGA | 12 | 0.9471 | 0.9592 | 0.0658 | 7,236,000 B |

Interpretation:

- FedAvg + QGA improves Macro-F1 and Attack Recall over the 28-feature FedAvg baseline.
- QGA reduces feature count from 28 to 12.
- Bandwidth decreases from 8,710,560 B to 7,236,000 B for QGA variants.
- QIFA remains a research comparison path.

Generated detailed reports and figures were externalized in Phase 3.
