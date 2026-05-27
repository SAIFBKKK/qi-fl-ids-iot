# P16.7 Realtime Packet Window Agent

This package prepares a safe packet-window feature extraction prototype for the live lab.

Target path:

`local packet observation -> packet window -> 28 features -> optional scaler -> optional QGA mask -> MQTT payload -> final IDS pipeline`

## Safety Defaults

- Dry-run is the default behavior.
- Synthetic packets are used by default.
- Local pcap input is supported when a local file is supplied.
- Live observation is disabled unless `--allow-live-capture` is passed explicitly.
- The package does not generate active traffic and does not run lab scenarios.

## Modules

- `packet_capture.py`: synthetic, optional local pcap, and explicitly gated live packet sources.
- `flow_window.py`: fixed-size `PacketWindow` implementation.
- `feature_extractor.py`: 28-feature CICIoT2023-like approximation layer.
- `scaler_runtime.py`: optional L1 scaler loading and final QGA mask application.
- `mqtt_runtime.py`: controlled MQTT JSON payload publisher.
- `edge_inference.py`: placeholder for future VM2 edge inference.
- `run_realtime_window_agent.py`: CLI entry point.

## Dry-Run Examples

```bash
python experiments/live_lab/realtime_agent/run_realtime_window_agent.py \
  --node-id iot-rpi-weak \
  --input-mode selected_12_scaled \
  --window-size 30 \
  --max-windows 1 \
  --dry-run
```

```bash
python experiments/live_lab/realtime_agent/run_realtime_window_agent.py \
  --node-id iot-smart-watch-medium \
  --input-mode original_28_scaled \
  --window-size 30 \
  --max-windows 1 \
  --dry-run
```

## Output Modes

- `original_28_unscaled`: extracted 28-feature vector without scaler.
- `original_28_scaled`: 28-feature vector after optional scaler; dry-run can continue without scaling if unavailable.
- `selected_12_scaled`: optional scaler followed by final QGA mask `conservative_seed_42`.
- `--no-scale`: explicit fallback switch for dry-runs when the scaler runtime cannot be used.

## Limits

The feature values are safe approximations for live-lab plumbing. They do not claim equivalence with the original CICIoT2023 feature extraction pipeline.
