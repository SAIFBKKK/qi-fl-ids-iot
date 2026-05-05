# E6 per-class summary

- Experiment: `exp_bench30_absent_qifa_28f`
- Scenario: `absent_local`
- Features: `28f`

## Top 5 worst F1 classes

- 4 DDoS-ACK_Fragmentation: F1=0.0001, support=17196
- 24 Mirai-greip_flood: F1=0.0002, support=44798
- 29 Recon-PortScan: F1=0.0002, support=44550
- 13 DDoS-TCP_Flood: F1=0.0364, support=20407
- 21 DoS-UDP_Flood: F1=0.0407, support=6743

## Top 5 best F1 classes

- 9 DDoS-RSTFINFlood: F1=0.9990, support=4967
- 15 DDoS-UDP_Fragmentation: F1=0.9874, support=26214
- 14 DDoS-UDP_Flood: F1=0.9207, support=39532
- 32 VulnerabilityScan: F1=0.9156, support=44678
- 8 DDoS-PSHACK_Flood: F1=0.8887, support=2405

## Rare classes summary

- No class has support below 1000 in this evaluation set.

## Brief interpretation

Low-F1 rows should be read together with support. Undefined values are written as NaN when a denominator is zero, so absent classes are not converted into artificial zeros.
