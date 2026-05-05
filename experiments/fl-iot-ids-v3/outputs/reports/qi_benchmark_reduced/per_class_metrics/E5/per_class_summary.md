# E5 per-class summary

- Experiment: `exp_bench30_absent_fedavg_28f`
- Scenario: `absent_local`
- Features: `28f`

## Top 5 worst F1 classes

- 29 Recon-PortScan: F1=0.0004, support=44550
- 4 DDoS-ACK_Fragmentation: F1=0.0010, support=17196
- 3 CommandInjection: F1=0.1246, support=29564
- 0 Backdoor_Malware: F1=0.1752, support=4592
- 16 DNS_Spoofing: F1=0.2850, support=45000

## Top 5 best F1 classes

- 9 DDoS-RSTFINFlood: F1=0.9993, support=4967
- 15 DDoS-UDP_Fragmentation: F1=0.9870, support=26214
- 12 DDoS-SynonymousIP_Flood: F1=0.9688, support=45000
- 14 DDoS-UDP_Flood: F1=0.9500, support=39532
- 18 DoS-HTTP_Flood: F1=0.9422, support=45000

## Rare classes summary

- No class has support below 1000 in this evaluation set.

## Brief interpretation

Low-F1 rows should be read together with support. Undefined values are written as NaN when a denominator is zero, so absent classes are not converted into artificial zeros.
