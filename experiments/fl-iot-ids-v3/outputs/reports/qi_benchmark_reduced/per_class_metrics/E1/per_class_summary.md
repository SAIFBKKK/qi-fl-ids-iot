# E1 per-class summary

- Experiment: `exp_bench30_normal_fedavg_28f`
- Scenario: `normal_noniid`
- Features: `28f`

## Top 5 worst F1 classes

- 31 Uploading_Attack: F1=0.2091, support=9390
- 0 Backdoor_Malware: F1=0.3528, support=22500
- 29 Recon-PortScan: F1=0.4181, support=45000
- 27 Recon-OSScan: F1=0.4458, support=45000
- 28 Recon-PingSweep: F1=0.4495, support=16965

## Top 5 best F1 classes

- 9 DDoS-RSTFINFlood: F1=0.9994, support=45000
- 6 DDoS-ICMP_Flood: F1=0.9991, support=45000
- 8 DDoS-PSHACK_Flood: F1=0.9964, support=45000
- 20 DoS-TCP_Flood: F1=0.9956, support=45000
- 13 DDoS-TCP_Flood: F1=0.9953, support=45000

## Rare classes summary

- No class has support below 1000 in this evaluation set.

## Brief interpretation

Low-F1 rows should be read together with support. Undefined values are written as NaN when a denominator is zero, so absent classes are not converted into artificial zeros.
