# E4 per-class summary

- Experiment: `exp_bench30_normal_qifa_qga15`
- Scenario: `normal_noniid`
- Features: `QGA-15`

## Top 5 worst F1 classes

- 0 Backdoor_Malware: F1=0.1156, support=22500
- 31 Uploading_Attack: F1=0.1569, support=9390
- 27 Recon-OSScan: F1=0.2301, support=45000
- 28 Recon-PingSweep: F1=0.2364, support=16965
- 16 DNS_Spoofing: F1=0.3503, support=45000

## Top 5 best F1 classes

- 6 DDoS-ICMP_Flood: F1=0.9992, support=45000
- 13 DDoS-TCP_Flood: F1=0.9975, support=45000
- 20 DoS-TCP_Flood: F1=0.9950, support=45000
- 25 Mirai-udpplain: F1=0.9942, support=45000
- 14 DDoS-UDP_Flood: F1=0.9927, support=45000

## Rare classes summary

- No class has support below 1000 in this evaluation set.

## Brief interpretation

Low-F1 rows should be read together with support. Undefined values are written as NaN when a denominator is zero, so absent classes are not converted into artificial zeros.
