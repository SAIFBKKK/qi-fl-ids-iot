# P9 QIFA Ablation

| method | features_count | macro_f1 | attack_recall | fpr | accuracy | aggregation_type | variant | gamma |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P5 FedAvg baseline | 28 | 0.9407415103639889 | 0.9474141414141414 | 0.06626666666666667 | 0.9408994708994709 | FedAvg |  |  |
| P8 FedAvg + QGA Flower | 12 | 0.9479843808395867 | 0.955010101010101 | 0.059444444444444446 | 0.9481269841269842 | FedAvg+QGA |  |  |
| P9 QIFA Flower | 28 | 0.9454121205767887 | 0.9436161616161616 | 0.0524 | 0.9455132275132275 | QIFA-Hybrid | hybrid | 0.5 |
| P9 QIFA + QGA Flower | 12 | 0.9471048783725642 | 0.9592121212121212 | 0.06584444444444444 | 0.9472804232804233 | QIFA-Hybrid | hybrid | 0.5 |
