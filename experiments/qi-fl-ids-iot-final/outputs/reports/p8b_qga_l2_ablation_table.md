# P8-b QGA L2 Ablation

Generated from latest available summaries.

| method | features_count | macro_f1 | weighted_f1 | macro_recall | macro_fpr | accuracy | model_size_bytes | bandwidth_total_bytes | gap_macro_f1_vs_p6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P6 L2 Flower baseline | 28 | 0.6355288381246096 | 0.7076571003455178 | 0.7294272787434922 | 0.04394011161971505 | 0.6841468381282211 | 49952 | 8991360 |  |
| P8-b L2 FedAvg + QGA Flower | 19 | 0.646557348155891 | 0.7762249172562404 | 0.7172291523344254 | 0.03316218834949124 | 0.7555178566719699 | 45344 | 8161920 | 0.011028510031281469 |
