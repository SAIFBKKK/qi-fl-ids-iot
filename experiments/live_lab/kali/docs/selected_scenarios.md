# Selected CICIoT2023-Inspired Scenarios

P16.10 documents the selected future scenarios only. No scenario is executed in this phase, and this document intentionally contains no executable command line.

| Scenario | CICIoT2023 family | Tool cited as scientific reference | Expected packet pattern | Expected features | Evidence to observe later |
|---|---|---|---|---|---|
| `icmp_flood_like` | DoS/DDoS | hping3 | IP plus ICMP-dominant windows | `ICMP`, `Rate`, `IAT`, `Number` | controlled feature window, MQTT flow, IDS prediction, IDS alert, dashboard event |
| `tcp_syn_recon_like` | Recon / DoS SYN | nmap / hping3 | IP plus TCP windows with SYN-dominant behavior | `TCP`, `syn_flag_number`, `syn_count`, `Rate` | controlled feature window, bridge metric, prediction topic, alert topic |
| `http_slow_like` | Web-Based / HTTP Flood | slowloris / golang-httpflood | TCP plus HTTP request-oriented windows | `HTTP`, `TCP`, `Duration`, `IAT`, `Rate` | controlled feature window, online-validator summary, IDS alert, dashboard timeline |

## Safety Boundary

These scenarios are reserved for a future controlled observation design. P16.10 only prepares inventory and documentation. The live defense demonstration remains based on controlled PacketWindow(30) MQTT publication.
