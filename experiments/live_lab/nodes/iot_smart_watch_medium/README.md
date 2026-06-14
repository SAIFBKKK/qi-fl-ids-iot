# iot-smart-watch-medium

P16.17 continuous smartwatch observer node.

This VM acts as a wearable IoT medium node in the live lab:

- registers as `smart_watch` and receives the `medium` tier;
- observes host-only packet metadata passively;
- builds sliding `PacketWindow(30)` windows;
- publishes `original_28_scaled` MQTT flow payloads;
- publishes status and window progress for the dashboard `/demo`.

Registration:

```bash
python3 experiments/live_lab/nodes/iot_smart_watch_medium/register_smartwatch_node.py \
  --controller-url http://192.168.56.1:8020
```

Continuous passive observer:

```bash
./experiments/live_lab/nodes/iot_smart_watch_medium/start_continuous_observer.sh
```

Dry-run simulation for local validation only:

```bash
python3 experiments/live_lab/nodes/iot_smart_watch_medium/smartwatch_passive_agent.py \
  --scenario icmp-lowrate-sim \
  --window-size 30 \
  --dry-run
```

The agent does not generate traffic, does not write pcap files, and does not
store application payloads.
