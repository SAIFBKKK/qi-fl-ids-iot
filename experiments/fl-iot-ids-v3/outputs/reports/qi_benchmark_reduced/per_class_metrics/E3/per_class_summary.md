# E3 per-class summary

- Experiment: `exp_bench30_normal_fedavg_qga15`
- Scenario: `normal_noniid`
- Features: `QGA-15`

## Top 5 worst F1 classes

- 0 Backdoor_Malware: F1=0.1143, support=22500
- 31 Uploading_Attack: F1=0.1646, support=9390
- 27 Recon-OSScan: F1=0.2527, support=45000
- 28 Recon-PingSweep: F1=0.2529, support=16965
- 17 DictionaryBruteForce: F1=0.3461, support=45000

## Top 5 best F1 classes

- 6 DDoS-ICMP_Flood: F1=0.9993, support=45000
- 13 DDoS-TCP_Flood: F1=0.9981, support=45000
- 20 DoS-TCP_Flood: F1=0.9962, support=45000
- 25 Mirai-udpplain: F1=0.9940, support=45000
- 9 DDoS-RSTFINFlood: F1=0.9924, support=45000

## Rare classes summary

- No class has support below 1000 in this evaluation set.

## Brief interpretation

Low-F1 rows should be read together with support. Undefined values are written as NaN when a denominator is zero, so absent classes are not converted into artificial zeros.
