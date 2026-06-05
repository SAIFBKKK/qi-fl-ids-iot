# P16.17 - Continuous Smartwatch Traffic Observer Report

## Objective

P16.17 prepares `iot-smart-watch-medium` for a continuous live defense demonstration. It adds a dedicated smartwatch observer that mirrors the drone P16.16 quality level: sliding `PacketWindow(30)`, status/window MQTT topics, 28 scaled features, dashboard presentation, and JSON-lines runtime evidence.

## Runtime Context

- Server Windows: `192.168.56.1`
- Smartwatch VM: `iot-smart-watch-medium`, `192.168.56.102`
- Kali lab workstation: `lab-attacker-kali`, `192.168.56.103`
- MQTT broker: `192.168.56.1:1883`
- Dashboard: `http://192.168.56.1:8013/demo`

## Components Added

- `experiments/live_lab/nodes/iot_smart_watch_medium/node_profile.yaml`
- `experiments/live_lab/nodes/iot_smart_watch_medium/register_smartwatch_node.py`
- `experiments/live_lab/nodes/iot_smart_watch_medium/smartwatch_passive_agent.py`
- `experiments/live_lab/nodes/iot_smart_watch_medium/start_continuous_observer.sh`
- `experiments/live_lab/docs/p16_17_continuous_smartwatch_observer.md`
- `experiments/live_lab/tests/test_smartwatch_pipeline.py`

## Validated Observation Summary

For the smartwatch node, two ICMP traffic profiles were generated from the Kali VM: a controlled low-rate ICMP reference flow and an accelerated ICMP flood-like pattern. The passive observer on the smartwatch captured PacketWindow(30), extracted 28 scaled CICIoT2023-like features, and published them to the MQTT-to-IDS pipeline. The final IDS API processed both flows without runtime errors and the dashboard displayed the corresponding attack predictions and critical alerts.

Observed counters:

- `ids/flows/iot-smart-watch-medium = 5`
- `ids/predictions/iot-smart-watch-medium = 5`
- `ids/alerts/iot-smart-watch-medium = 5`
- `final_ids_api_prediction_errors_total = 0`
- Low-rate ICMP: `Rate scaled ~= -0.3507`
- Accelerated ICMP flood-like: `Rate scaled ~= 1.3841`

## Rate Scaled Explanation

Rate scaled is the RobustScaler-normalized version of the raw packet rate. It expresses the position of the current traffic window relative to the training distribution rather than the raw packet-per-second value. In the validated smartwatch test, the low-rate ICMP reference flow produced a lower scaled Rate value, while the accelerated ICMP flood-like pattern produced a clearly higher scaled Rate value, confirming that the observer captured two distinct traffic profiles.

## Feature Consistency

For ICMP-only packet windows:

- `Protocol Type` maps to ICMP.
- `ICMP` reflects the count of ICMP packets in the window.
- `TCP = 0`.
- `UDP = 0`.
- `syn_flag_number = 0`.
- `syn_count = 0`.
- `Number = 30` for a complete window.

## Dashboard Changes

The `/demo` dashboard now includes:

- Smartwatch Live Observer panel;
- PacketWindow fill and progress bar;
- scaled Rate and scaled IAT values;
- ICMP/TCP/UDP counts;
- runtime error badge;
- Smartwatch ICMP Feature Contrast table;
- Smartwatch Pipeline panel.

## Safety Boundaries

P16.17 does not add traffic-generation scripts. Passive mode only observes metadata on the host-only lab network. Simulation modes generate packet metadata in memory only. No pcap, dataset, checkpoint, or large runtime log is committed.

## Remaining Limits

The feature extractor is still a CICIoT2023-like deployment prototype. It is useful for live runtime evidence and demonstration, but it does not replace the scientific P12/P13 evaluation.
