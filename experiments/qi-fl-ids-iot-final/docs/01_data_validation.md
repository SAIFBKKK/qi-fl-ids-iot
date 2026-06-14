# P1 — Data Validation Report

## 1. Objectif

Valider le dataset final CIC-IoT avant preprocessing, split, scaling ou entraînement.

## 2. Dataset utilisé

- Parquet prioritaire : `data/balancing_v3_fixed300k_outputs/balancing_v3_fixed300k_balanced.parquet`
- CSV secondaire : `data/balancing_v3_fixed300k_outputs/balancing_v3_fixed300k_balanced.csv`
- Mapping labels : `data/balancing_v3_fixed300k_outputs/label_mapping.json`

## 3. Méthode de validation

La validation utilise les métadonnées Parquet puis une lecture par row groups et colonnes. Le CSV complet n'est pas chargé.

## 4. Schéma du dataset

- Shape : `9401350 x 29`
- Row groups : `9`
- Colonne label : `label_id`

## 5. Features validées

- Nombre de features : `28`
- Features numériques : `True`
- `label_id` exclu des features : `True`

## 6. Mapping des labels

- Classes : `34`
- `BenignTraffic` = `1` : `True`
- Label ids inconnus : `[]`

## 7. Mapping binaire L1

Pour L1, `normal = 0` et `attack = 1`. Le `label_id` original `1` (`BenignTraffic`) devient `binary_label = 0`; toutes les autres classes deviennent `binary_label = 1`.

| name | binary_label | count | ratio |
| --- | --- | --- | --- |
| normal | 0 | 300000 | 0.031910 |
| attack | 1 | 9101350 | 0.968090 |

## 8. Mapping familles L2

| family | count | ratio |
| --- | --- | --- |
| Benign | 300000 | 0.031910 |
| BruteForce | 300000 | 0.031910 |
| DDoS | 3600000 | 0.382924 |
| DoS | 1200000 | 0.127641 |
| Malware | 150000 | 0.015955 |
| Mirai | 900000 | 0.095731 |
| Recon | 1313100 | 0.139671 |
| Spoofing | 600000 | 0.063821 |
| Web-based | 1038250 | 0.110436 |

## 9. Distribution 34 classes L3

| label_id | label | count | ratio |
| --- | --- | --- | --- |
| 0 | Backdoor_Malware | 150000 | 0.015955 |
| 1 | BenignTraffic | 300000 | 0.031910 |
| 2 | BrowserHijacking | 292950 | 0.031160 |
| 3 | CommandInjection | 270450 | 0.028767 |
| 4 | DDoS-ACK_Fragmentation | 300000 | 0.031910 |
| 5 | DDoS-HTTP_Flood | 300000 | 0.031910 |
| 6 | DDoS-ICMP_Flood | 300000 | 0.031910 |
| 7 | DDoS-ICMP_Fragmentation | 300000 | 0.031910 |
| 8 | DDoS-PSHACK_Flood | 300000 | 0.031910 |
| 9 | DDoS-RSTFINFlood | 300000 | 0.031910 |
| 10 | DDoS-SYN_Flood | 300000 | 0.031910 |
| 11 | DDoS-SlowLoris | 300000 | 0.031910 |
| 12 | DDoS-SynonymousIP_Flood | 300000 | 0.031910 |
| 13 | DDoS-TCP_Flood | 300000 | 0.031910 |
| 14 | DDoS-UDP_Flood | 300000 | 0.031910 |
| 15 | DDoS-UDP_Fragmentation | 300000 | 0.031910 |
| 16 | DNS_Spoofing | 300000 | 0.031910 |
| 17 | DictionaryBruteForce | 300000 | 0.031910 |
| 18 | DoS-HTTP_Flood | 300000 | 0.031910 |
| 19 | DoS-SYN_Flood | 300000 | 0.031910 |
| 20 | DoS-TCP_Flood | 300000 | 0.031910 |
| 21 | DoS-UDP_Flood | 300000 | 0.031910 |
| 22 | MITM-ArpSpoofing | 300000 | 0.031910 |
| 23 | Mirai-greeth_flood | 300000 | 0.031910 |
| 24 | Mirai-greip_flood | 300000 | 0.031910 |
| 25 | Mirai-udpplain | 300000 | 0.031910 |
| 26 | Recon-HostDiscovery | 300000 | 0.031910 |
| 27 | Recon-OSScan | 300000 | 0.031910 |
| 28 | Recon-PingSweep | 113100 | 0.012030 |
| 29 | Recon-PortScan | 300000 | 0.031910 |
| 30 | SqlInjection | 262250 | 0.027895 |
| 31 | Uploading_Attack | 62600 | 0.006659 |
| 32 | VulnerabilityScan | 300000 | 0.031910 |
| 33 | XSS | 150000 | 0.015955 |

## 10. Valeurs manquantes

- Total NaN/null : `0`

## 11. Valeurs infinies

- Total ±inf : `0`

## 12. Statistiques des features

| feature | min | max | mean | std |
| --- | --- | --- | --- | --- |
| flow_duration | 0.0 | 186698.3051801586 | 114.89312542904021 | 1122.9048107930787 |
| Header_Length | 0.0 | 9907147.75 | 236188.49649985338 | 800636.5853682298 |
| Protocol Type | 0.0 | 47.0 | 10.534018184897757 | 10.143895754456087 |
| Duration | 0.0 | 255.0 | 78.36565437997642 | 29.661102636428627 |
| Rate | 0.0 | 8388608.0 | 5881.701857011653 | 70475.74596136274 |
| fin_flag_number | 0.0 | 1.0 | 0.03246639451188695 | 0.17676569736692105 |
| syn_flag_number | 0.0 | 1.0 | 0.12224248111615338 | 0.3209471062182823 |
| rst_flag_number | 0.0 | 1.0 | 0.05287197208623737 | 0.21653048962164195 |
| psh_flag_number | 0.0 | 1.0 | 0.05143989194067979 | 0.21284788395270207 |
| ack_flag_number | 0.0 | 1.0 | 0.3702852307821769 | 0.46385520995143925 |
| ack_count | 0.0 | 7.4 | 0.08147075075576757 | 0.22296395788643733 |
| syn_count | 0.0 | 12.87 | 0.589061619792118 | 0.7807290274599701 |
| fin_count | 0.0 | 164.73 | 0.11504805510936468 | 0.45026811473757844 |
| urg_count | 0.0 | 4401.7 | 37.372492407612 | 161.89096202199374 |
| rst_count | 0.0 | 9613.0 | 167.12580964071768 | 604.3358181874121 |
| HTTP | 0.0 | 1.0 | 0.09851306448065976 | 0.28373650058458877 |
| HTTPS | 0.0 | 1.0 | 0.16131795454940734 | 0.3462430839765482 |
| DNS | 0.0 | 1.0 | 0.0014827013106658551 | 0.03349024478802421 |
| SSH | 0.0 | 1.0 | 0.004705162660168334 | 0.060600216113411494 |
| TCP | 0.0 | 1.0 | 0.612307350555971 | 0.4727382627175389 |
| UDP | 0.0 | 1.0 | 0.1572647926824488 | 0.3525709791751592 |
| ARP | 0.0 | 1.0 | 0.0017083057225528685 | 0.03622682259649227 |
| ICMP | 0.0 | 1.0 | 0.06308894813568018 | 0.24311688407286078 |
| Tot sum | 50.0 | 127335.8 | 3593.9841451360794 | 4720.817194654835 |
| Min | 42.0 | 11650.0 | 169.78027437109156 | 238.8284039383311 |
| Std | 0.0 | 12385.2391048564 | 180.7791143822679 | 295.84712509254734 |
| IAT | 0.0 | 167639436.0415293 | 83299299.80002978 | 45701725.356863104 |
| Number | 1.0 | 15.0 | 9.495460561211098 | 2.199558160792378 |

## 13. Artefacts générés

- `experiments/qi-fl-ids-iot-final/outputs/artifacts/features/feature_names.json`
- `experiments/qi-fl-ids-iot-final/outputs/artifacts/mappings/label_mapping.json`
- `experiments/qi-fl-ids-iot-final/outputs/artifacts/mappings/id_to_label.json`
- `experiments/qi-fl-ids-iot-final/outputs/artifacts/mappings/label_to_family.json`
- `experiments/qi-fl-ids-iot-final/outputs/artifacts/mappings/label_to_binary.json`

## 14. Figures générées

- `experiments/qi-fl-ids-iot-final/outputs/figures/data_validation/01_binary_distribution.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/data_validation/02_class_distribution_34.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/data_validation/03_family_distribution.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/data_validation/04_missing_values_by_column.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/data_validation/05_infinite_values_by_column.png`
- `experiments/qi-fl-ids-iot-final/outputs/figures/data_validation/06_feature_types.png`

## 15. Risques restants

- Aucun warning bloquant.

## 16. Critères d’acceptation

| critere | ok |
| --- | --- |
| parquet_found | True |
| label_mapping_found | True |
| shape_matches | True |
| features_count_28 | True |
| label_id_present | True |
| numeric_features | True |
| classes_count_34 | True |
| benign_label_id | True |
| dataset_label_ids_known | True |
| missing_values_computed | True |
| infinite_values_computed | True |
| binary_distribution_generated | True |
| family_distribution_generated | True |
| class_distribution_generated | True |
| label_excluded_from_features | True |

## 17. Conclusion P1

P1 est validée.


