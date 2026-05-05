# E7 per-class summary

- Experiment: `exp_bench30_absent_fedavg_qga15`
- Scenario: `absent_local`
- Features: `QGA-15`

## Top 5 worst F1 classes

- 4 DDoS-ACK_Fragmentation: F1=0.0005, support=17196
- 15 DDoS-UDP_Fragmentation: F1=0.0008, support=26214
- 3 CommandInjection: F1=0.0011, support=29564
- 0 Backdoor_Malware: F1=0.0238, support=4592
- 25 Mirai-udpplain: F1=0.0292, support=43631

## Top 5 best F1 classes

- 32 VulnerabilityScan: F1=0.7932, support=44678
- 18 DoS-HTTP_Flood: F1=0.6791, support=45000
- 5 DDoS-HTTP_Flood: F1=0.6695, support=45000
- 10 DDoS-SYN_Flood: F1=0.6678, support=45000
- 11 DDoS-SlowLoris: F1=0.6485, support=44800

## Rare classes summary

- No class has support below 1000 in this evaluation set.

## Brief interpretation

Low-F1 rows should be read together with support. Undefined values are written as NaN when a denominator is zero, so absent classes are not converted into artificial zeros.
