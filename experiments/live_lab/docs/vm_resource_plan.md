# VM Resource Plan

Target PC total RAM: `16 GB`.

Keep Windows and Docker Desktop as the priority. The VMs are intentionally small
so the server stack remains responsive.

## Recommended Allocation

| VM | vCPU | RAM | Disk |
|---|---:|---:|---:|
| `iot-rpi-weak` | 1 | 1024 MB | 12 GB |
| `iot-smart-watch-medium` | 1 | 1536 MB | 15 GB |
| `lab-attacker-kali` | 1 | 1536 MB | 20 GB |

Total VM RAM: `4096 MB`.

## If the PC Becomes Slow

- Stop Grafana when visual evidence is not being collected.
- Stop one VM not used in the current demo segment.
- Reduce `iot-smart-watch-medium` to `1024 MB`.
- Start only two VMs at a time.
- Keep Kali powered off outside demonstration windows.

