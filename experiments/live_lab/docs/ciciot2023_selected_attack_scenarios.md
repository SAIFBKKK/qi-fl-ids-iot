# CICIoT2023 Selected Scenario Mapping

The following scenarios are documented as safe, controlled, CICIoT2023-inspired
feature-window demonstrations. Tool names are scientific references only. This
file intentionally contains no executable commands.

| Scenario | CICIoT2023 family | Tool cited in CICIoT2023 | Expected packet structure | Impacted 28 features | Safe lab implementation | Evidence to collect |
|---|---|---|---|---|---|---|
| `icmp_flood_like` | DoS/DDoS | hping3 reference | IP + ICMP | `ICMP`, `Rate`, `IAT`, `Number`, `Header_Length`, `Tot sum`, `Min`, `Std` | Controlled replay, local controlled pcap, or feature windows | feature JSON/CSV, MQTT prediction, alert topic, dashboard screenshot |
| `tcp_syn_recon_like` | Recon / DoS SYN | nmap and hping3 references | IP + TCP with SYN-dominant windows | `TCP`, `syn_flag_number`, `syn_count`, `Rate`, `IAT`, `Number`, `Header_Length` | Controlled replay, local controlled pcap, or feature windows | feature JSON/CSV, MQTT prediction, bridge metrics, validator summary |
| `http_slow_like` | Web-Based / HTTP Flood | slowloris and golang-httpflood references | TCP + HTTP requests | `HTTP`, `TCP`, `Duration`, `Rate`, `IAT`, `psh_flag_number`, `ack_count`, `Tot sum` | Controlled replay, local controlled pcap, or feature windows | feature JSON/CSV, prediction topic, alert topic, Grafana panel |

No scenario in P16.3 is executed. This is a planning and documentation step.

