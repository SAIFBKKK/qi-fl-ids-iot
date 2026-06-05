# P16.7 Realtime Feature Extraction Prototype

## Objective

P16.7 prepares a passive packet-window feature extraction prototype for the live lab.

Target path:

`local packet observation -> packet window -> 28 CICIoT2023-like features -> optional scaler -> optional QGA mask -> MQTT payload -> final IDS pipeline`

## Difference From P16.6

- P16.6 validated controlled MQTT JSON payloads from VM nodes to final IDS predictions and alerts.
- P16.7 prepares the next layer: deriving those payloads from packet windows.

## Window Design

- Default window size: `30` packets.
- Default source: synthetic dry-run packets.
- Local pcap input: supported only when a local pcap path is provided.
- Live observation: disabled unless `--allow-live-capture` is explicitly passed.

## Input Modes

- `original_28_unscaled`: emit the 28 extracted values as-is.
- `original_28_scaled`: apply the optional L1 scaler when available; dry-run continues with a clear warning if scaling cannot run.
- `selected_12_scaled`: apply the optional scaler and then the final QGA mask `conservative_seed_42`.
- `--no-scale`: explicit fallback switch for dry-runs when scaler execution is not desired.

## Feature Approximation Limits

The extractor preserves the project feature order:

`flow_duration, Header_Length, Protocol Type, Duration, Rate, fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number, ack_flag_number, ack_count, syn_count, fin_count, urg_count, rst_count, HTTP, HTTPS, DNS, SSH, TCP, UDP, ARP, ICMP, Tot sum, Min, Std, IAT, Number`

P16.7 computes safe CICIoT2023-like approximations from normalized packets. It does not claim equivalence with the original CICIoT2023 extractor.

## VM1 Dry-Run

```bash
cd ~/qi-fl-ids-iot
git pull
python3 experiments/live_lab/realtime_agent/run_realtime_window_agent.py \
  --node-id iot-drone-sitl \
  --broker 192.168.56.1 \
  --input-mode selected_12_scaled \
  --window-size 30 \
  --max-windows 1 \
  --dry-run
```

## VM2 Dry-Run

```bash
cd ~/qi-fl-ids-iot
git pull
python3 experiments/live_lab/realtime_agent/run_realtime_window_agent.py \
  --node-id iot-smart-watch-medium \
  --broker 192.168.56.1 \
  --input-mode original_28_scaled \
  --window-size 30 \
  --max-windows 1 \
  --dry-run
```

## Controlled Publish Later

After dry-run validation, add `--publish` to publish the generated JSON payload to:

`ids/flows/{node_id}`

Do this only when the server stack and observers are ready.

## Scope Reminder

P16.7 does not start Kali scenarios, does not generate active traffic, does not train models, and does not modify P8-P16 results.

