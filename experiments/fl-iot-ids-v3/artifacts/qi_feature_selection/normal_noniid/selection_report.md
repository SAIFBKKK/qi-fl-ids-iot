# QGA Feature Selection Report

- Scenario: normal_noniid
- Input features: 28
- Selected features: 15
- Mode: smoke
- Generations: 3
- Population size: 4
- Mini-MLP epochs: 1
- Best validation Macro-F1: 0.017974
- Best fitness: 0.017974

## Selected Features

1. `2` - `Protocol Type`
2. `3` - `Duration`
3. `6` - `syn_flag_number`
4. `7` - `rst_flag_number`
5. `8` - `psh_flag_number`
6. `9` - `ack_flag_number`
7. `10` - `ack_count`
8. `12` - `fin_count`
9. `13` - `urg_count`
10. `16` - `HTTPS`
11. `20` - `UDP`
12. `21` - `ARP`
13. `23` - `Tot sum`
14. `24` - `Min`
15. `27` - `Number`

This selector is quantum-inspired: it maintains a theta vector of
length 28, samples masks with p_i = sin(theta_i)^2, repairs each mask
to exactly K selected features, and updates theta toward the best mask.
It does not use quantum hardware.
