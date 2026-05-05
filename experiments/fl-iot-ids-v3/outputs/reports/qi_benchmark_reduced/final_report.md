# Reduced QI Benchmark Final Report

## 1. Objective
Compare a focused 8-experiment FL matrix plus one centralized reference for the QI sprint.

## 2. Experimental Design
Two scenarios (`normal_noniid`, `absent_local`), two aggregations (FedAvg, QIFA), and two feature settings (28 features, QGA-15).

## 3. Benchmark Matrix (E1-E8 + R)

| ID | Scenario | Aggregation | Features | Status |
| --- | --- | --- | --- | --- |
| E1 | normal_noniid | FedAvg | 28 | success_30_rounds |
| E2 | normal_noniid | QIFA | 28 | success_30_rounds |
| E3 | normal_noniid | FedAvg | QGA-15 | success_30_rounds |
| E4 | normal_noniid | QIFA | QGA-15 | success_30_rounds |
| E5 | absent_local | FedAvg | 28 | success_30_rounds |
| E6 | absent_local | QIFA | 28 | success_30_rounds |
| E7 | absent_local | FedAvg | QGA-15 | success_30_rounds |
| E8 | absent_local | QIFA | QGA-15 | success_30_rounds |
| R | normal_noniid | Centralized | 28 | success |

## 4. Dataset Scenarios
`normal_noniid` is the moderate non-IID scenario. `absent_local` removes local class coverage and should increase client diversity.

## 5. Methods
FedAvg is the classical baseline. QIFA applies normalized diversity-aware client weighting. QGA-15 uses a theta-vector quantum-inspired selector over the real 28-feature input.

## 6. QIFA Formulation
`epsilon_k = ||w_k - w_avg|| / (||w_avg|| + 1e-8)` and effective client weights are normalized before aggregation.

## 7. QGA Feature-Selection Protocol
The serious protocol is configured with `k_features=15`, `n_generations=20`, `pop_size=15`, `epochs=2`, `max_samples_per_class=2000`, `seed=42`.

## 8. Results Tables
See `final_comparison_table.csv`. E1-E8 are completed 30-round FL runs with `status=success`; the centralized reference R is included for context.

## 9. Figure 1 Analysis
Macro-F1 convergence is plotted when 30-round round metrics exist.

## 10. Figure 2 Analysis
Loss convergence is plotted when distributed loss curves exist.

## 11. Figure 3 Analysis
Final barplot compares available final metrics for E1-E8 and R.

## 12. Figure 4 Analysis
Confusion matrices for E1, E2, E5, and E6 were generated from the completed 30-round best checkpoints using `evaluate_confusion_matrices.py`.

## 13. Figure 5 Analysis
QIFA diversity curves compare E2 and E6 when both QIFA runs are available.

## 14. Figure 6 Analysis
Bandwidth figure is generated only if all QGA-15 comparison runs have valid communication metrics.

## 15. Comparison With Centralized Reference R
R comes from `outputs/model_factory_30rounds` and is centralized/model-factory validation, not an identical FL protocol. Interpret comparisons accordingly.

## 16. Discussion
The reduced benchmark now uses the completed E1-E8 30-round runs. Earlier 3-round smoke or partial runs are not used as final benchmark results.

## 17. Limitations
The conclusions are limited to the reduced 8-run matrix and the centralized reference. Windows/Ray execution can be slow on CPU. Centralized R is a useful reference but not protocol-identical.

## 18. Final Conclusion
Completed/reusable rows currently available: 9 / 9. No fabricated results are reported.

## Final Ablation Interpretation

### Lot 1: QIFA effect on 28 features

- normal_noniid, E2 - E1: delta Macro-F1 `-0.002467134906848578`, delta accuracy `-0.00222095684096546`, delta FPR `0.01910214557762248`, delta rare recall `-0.0020484995385492977`.
- absent_local, E6 - E5: delta Macro-F1 `-0.01777445656660992`, delta accuracy `-0.012020414267692958`, delta FPR `-0.0029526606236371794`, delta rare recall `0.01671818515048018`.

Interpretation: on 28 features, QIFA does not improve final Macro-F1. It is slightly worse on normal_noniid and more clearly lower on absent_local. On absent_local it improves FPR and rare-class metrics, which is a useful IDS trade-off rather than a global win.

### Lot 2: QGA-15 feature-selection effect

- FedAvg normal_noniid, E3 - E1: delta Macro-F1 `-0.06522166627854042`, delta update bytes `-39936.0`.
- QIFA normal_noniid, E4 - E2: delta Macro-F1 `-0.07369079253798139`, delta update bytes `-39936.0`.
- FedAvg absent_local, E7 - E5: delta Macro-F1 `-0.2298100850308853`, delta update bytes `-39936.0`.
- QIFA absent_local, E8 - E6: delta Macro-F1 `-0.20014446360418475`, delta update bytes `-39936.0`.

Interpretation: QGA-15 reduces update payload size from `536472.0` to `496536.0` bytes in these runs, but it degrades Macro-F1 in all four direct comparisons. It is therefore a communication/compactness trade-off, not a quality improvement in this benchmark.

### Full QI stack: QGA-15 + QIFA interaction

- normal_noniid: best classical `E1` with Macro-F1 `0.7697173983054993`; best QI `E2` with Macro-F1 `0.7672502633986508`; full QI gain vs best classical `-0.07615792744482996`. Verdict: Classical remains best on Macro-F1; inspect QI trade-offs.
- absent_local: best classical `E5` with Macro-F1 `0.49375834494747295`; best QI `E6` with Macro-F1 `0.47598388838086303`; full QI gain vs best classical `-0.21791892017079467`. Verdict: Classical remains best on Macro-F1; inspect QI trade-offs.

The full QI stack does not beat the best classical Macro-F1 in either scenario. Its defensible contribution is more nuanced: QIFA improves some rare/FPR behavior under absent_local, while QGA-15 lowers communication cost with a performance penalty.

### Final conclusion

For this reduced 8-run benchmark, FedAvg with all 28 features remains the strongest Macro-F1 baseline. QIFA is most interesting under `absent_local`, where it improves FPR and rare-class recall despite lower Macro-F1. QGA-15 consistently reduces model update size but degrades predictive quality, so it should be reported as a compactness/communication trade-off. No missing metrics were filled artificially; unavailable values are left as `NaN` in the generated tables.
