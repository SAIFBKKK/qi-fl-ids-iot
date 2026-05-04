# QGA-15 Selected Features

Selection protocol:

- Input features: 28
- Selected features: 15
- Generations: 20
- Population size: 15
- Mini-MLP epochs: 2
- Max samples per class: 2000
- Seed: 42

## normal_noniid

- Best validation Macro-F1 used as selector fitness: `0.04006989496906574`
- Artifact: `artifacts/qi_feature_selection/normal_noniid/selected_features.json`

Selected features:

1. `Header_Length`
2. `Protocol Type`
3. `Duration`
4. `rst_flag_number`
5. `syn_count`
6. `fin_count`
7. `HTTP`
8. `DNS`
9. `TCP`
10. `UDP`
11. `ICMP`
12. `Tot sum`
13. `Std`
14. `IAT`
15. `Number`

## absent_local

- Best validation Macro-F1 used as selector fitness: `0.026517878725061478`
- Artifact: `artifacts/qi_feature_selection/absent_local/selected_features.json`

Selected features:

1. `flow_duration`
2. `Duration`
3. `syn_flag_number`
4. `rst_flag_number`
5. `fin_count`
6. `urg_count`
7. `rst_count`
8. `HTTP`
9. `HTTPS`
10. `DNS`
11. `SSH`
12. `TCP`
13. `UDP`
14. `ARP`
15. `Number`

These selector fitness values are only used to choose the feature masks. They
are not FL benchmark results.
