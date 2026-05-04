# QGA Feature Selection Report

- Scenario: absent_local
- Input features: 28
- Selected features: 15
- Mode: full
- Generations: 20
- Population size: 15
- Mini-MLP epochs: 2
- Best validation Macro-F1: 0.026518
- Best fitness: 0.026518

## Selected Features

1. `0` - `flow_duration`
2. `3` - `Duration`
3. `6` - `syn_flag_number`
4. `7` - `rst_flag_number`
5. `12` - `fin_count`
6. `13` - `urg_count`
7. `14` - `rst_count`
8. `15` - `HTTP`
9. `16` - `HTTPS`
10. `17` - `DNS`
11. `18` - `SSH`
12. `19` - `TCP`
13. `20` - `UDP`
14. `21` - `ARP`
15. `27` - `Number`

This selector is quantum-inspired: it maintains a theta vector of
length 28, samples masks with p_i = sin(theta_i)^2, repairs each mask
to exactly K selected features, and updates theta toward the best mask.
It does not use quantum hardware.
