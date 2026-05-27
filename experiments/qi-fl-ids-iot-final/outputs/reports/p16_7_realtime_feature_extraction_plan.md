# P16.7 Realtime Packet Window Feature Extraction Plan

## Objective

P16.7 prepares a safe real-time packet-window feature extraction prototype for the live lab.

Target architecture:

`local packet observation -> PacketWindow(30) -> 28 CICIoT2023-like features -> optional scaler -> optional QGA mask -> MQTT ids/flows/{node_id} -> final IDS pipeline`

## Components Created

- `packet_capture.py`: synthetic dry-run source, optional local pcap source, and explicitly gated live packet source.
- `flow_window.py`: fixed-size packet windows with summaries.
- `feature_extractor.py`: 28-feature extraction preserving the project schema order.
- `scaler_runtime.py`: optional scaler loading and `conservative_seed_42` QGA mask application.
- `mqtt_runtime.py`: controlled MQTT payload publication to `ids/flows/{node_id}`.
- `edge_inference.py`: placeholder for future medium-node edge inference.
- `run_realtime_window_agent.py`: CLI orchestration for dry-run and controlled publish.

## Feature Mapping

The extractor preserves the order from:

`experiments/qi-fl-ids-iot-final/outputs/artifacts/features/feature_names.json`

Fallback order:

`flow_duration, Header_Length, Protocol Type, Duration, Rate, fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number, ack_flag_number, ack_count, syn_count, fin_count, urg_count, rst_count, HTTP, HTTPS, DNS, SSH, TCP, UDP, ARP, ICMP, Tot sum, Min, Std, IAT, Number`

All P16.7 values are marked as approximations. Protocol counts, flags, lengths, duration, rate, inter-arrival time, and service counters are computed from the normalized packet representation.

## Gap With Full CICIoT2023 Processing

- P16.7 does not reproduce the full CICIoT2023 extractor.
- P16.7 does not claim scientific equivalence with CICIoT2023 CSV generation.
- Application features are inferred from ports, not deep protocol parsing.
- The scaler is optional at runtime and may fall back to no-scale mode for dry-runs.
- Edge inference remains a placeholder; server-side MQTT inference remains the validated path.

## Validation Expected

- `selected_12_scaled` dry-run emits 12 features.
- `original_28_scaled` dry-run emits 28 features.
- `original_28_unscaled` dry-run emits 28 features.
- No live observation is started unless explicitly requested.
- No active traffic is generated.
- `--no-scale` is available as an explicit dry-run fallback, but the default uses the scaler when it is loadable.

## Limits

- This is a prototype for live-lab plumbing and evidence collection.
- It does not change final model artifacts or P8-P16 results.
- It does not commit pcap files, datasets, logs, checkpoints, or run outputs.

## Next Step

P16.8 should validate controlled publication from packet-window output into the already validated P16.6 MQTT-to-IDS path.
