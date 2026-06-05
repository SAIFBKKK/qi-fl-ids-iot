# P16.17 - Continuous Smartwatch Traffic Observer

## Objective

P16.17 gives `iot-smart-watch-medium` the same runtime presentation quality as the drone node. The smartwatch VM can run a continuous passive observer, build sliding `PacketWindow(30)` windows, extract 28 CICIoT2023-like features, apply the runtime RobustScaler JSON, and publish `original_28_scaled` payloads to the existing MQTT-to-IDS pipeline.

## Architecture

```text
Kali VM 192.168.56.103
  -> benign or controlled lab traffic metadata observed passively
Smartwatch VM 192.168.56.102
  -> PacketWindow(30), stride 15
  -> 28 CICIoT2023-like features
  -> runtime scaler JSON
  -> MQTT ids/flows/iot-smart-watch-medium
Final MQTT bridge
  -> final IDS API P8 FedAvg + QGA
  -> ids/predictions/iot-smart-watch-medium
  -> ids/alerts/iot-smart-watch-medium
Dashboard /demo
```

The smartwatch publishes 28 scaled features. It does not apply the QGA mask on the VM. The final IDS API applies the mask as already validated in P16.6 to P16.8.1.

## MQTT Topics

- `ids/flows/iot-smart-watch-medium`
- `ids/status/iot-smart-watch-medium`
- `ids/windows/iot-smart-watch-medium`
- `ids/predictions/iot-smart-watch-medium`
- `ids/alerts/iot-smart-watch-medium`

## VM2 Commands

Register the node:

```bash
cd ~/qi-fl-ids-iot
python3 experiments/live_lab/nodes/iot_smart_watch_medium/register_smartwatch_node.py \
  --controller-url http://192.168.56.1:8020
```

Run the continuous passive observer:

```bash
cd ~/qi-fl-ids-iot
./experiments/live_lab/nodes/iot_smart_watch_medium/start_continuous_observer.sh
```

Local metadata-only validation:

```bash
python3 experiments/live_lab/nodes/iot_smart_watch_medium/smartwatch_passive_agent.py \
  --scenario icmp-lowrate-sim \
  --window-size 30 \
  --dry-run
```

## Server Verification

```bash
curl http://192.168.56.1:8015/summary | grep iot-smart-watch-medium
curl http://192.168.56.1:8016/metrics | grep iot-smart-watch-medium
curl http://192.168.56.1:8014/metrics
```

Dashboard:

```text
http://192.168.56.1:8013/demo
```

## Rate Scaled Explanation

`Rate scaled` is not the raw packet-per-second rate. It is the RobustScaler-normalized value:

```text
Rate_scaled = (Rate_raw - median_train) / scale_train
```

A negative value means the observed traffic window is slower than the learned training center. A positive high value means the observed traffic window is accelerated relative to that training distribution.

For ICMP windows:

- `ICMP = 30` for a full ICMP-only `PacketWindow(30)`;
- `TCP = 0`;
- `UDP = 0`;
- `syn_flag_number = 0`;
- `syn_count = 0`;
- `Number = 30`.

## Safety Scope

The repository does not include scripts that generate offensive traffic. The smartwatch agent is passive in live mode. Its simulation modes create local packet metadata only, do not send network packets, do not write pcap files, and do not store application payloads.

## Scientific Limits

The P16.17 observer is deployment evidence, not a scientific reproduction of CICIoT2023 traffic generation. Feature extraction remains a CICIoT2023-like prototype from packet metadata. Scientific evaluation remains the P12/P13 evaluation path.
