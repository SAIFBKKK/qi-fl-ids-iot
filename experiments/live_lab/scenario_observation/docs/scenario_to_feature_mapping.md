# Scenario To Feature Mapping

| scenario_id | CICIoT2023 family | reference tools | packet pattern | expected features | expected dashboard evidence |
|---|---|---|---|---|---|
| `icmp_flood_like` | DoS/DDoS | hping3 reference only | IP plus ICMP-dominant windows | `ICMP`, `Rate`, `IAT`, `Number` | scenario label, PacketWindow(30) flow, prediction, alert, zero errors |
| `tcp_syn_recon_like` | Recon / DoS SYN | nmap and hping3 references only | IP plus TCP windows with SYN-dominant behavior | `TCP`, `syn_flag_number`, `syn_count`, `Rate` | scenario label, target node, prediction, alert, zero errors |
| `http_slow_like` | Web-Based / HTTP Flood | slowloris and golang-httpflood references only | TCP plus HTTP request-oriented windows | `HTTP`, `TCP`, `Duration`, `IAT`, `Rate` | scenario label, latest PacketWindow flow, prediction, alert, zero errors |

## Notes

These mappings are expected live-lab tendencies, not a scientific reproduction of CICIoT2023 generation. The final scientific evaluation remains P12/P13.
