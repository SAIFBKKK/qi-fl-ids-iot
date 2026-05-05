# E8 per-class summary

- Experiment: `exp_bench30_absent_qifa_qga15`
- Scenario: `absent_local`
- Features: `QGA-15`

## Top 5 worst F1 classes

- 3 CommandInjection: F1=0.0003, support=29564
- 15 DDoS-UDP_Fragmentation: F1=0.0004, support=26214
- 4 DDoS-ACK_Fragmentation: F1=0.0009, support=17196
- 21 DoS-UDP_Flood: F1=0.0094, support=6743
- 25 Mirai-udpplain: F1=0.0267, support=43631

## Top 5 best F1 classes

- 32 VulnerabilityScan: F1=0.7672, support=44678
- 18 DoS-HTTP_Flood: F1=0.6766, support=45000
- 10 DDoS-SYN_Flood: F1=0.6757, support=45000
- 5 DDoS-HTTP_Flood: F1=0.6686, support=45000
- 11 DDoS-SlowLoris: F1=0.6531, support=44800

## Rare classes summary

- No class has support below 1000 in this evaluation set.

## Brief interpretation

Low-F1 rows should be read together with support. Undefined values are written as NaN when a denominator is zero, so absent classes are not converted into artificial zeros.
