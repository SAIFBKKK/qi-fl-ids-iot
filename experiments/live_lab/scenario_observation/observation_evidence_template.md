# Controlled Scenario Observation Evidence Template

## scenario_id

`<icmp_flood_like | tcp_syn_recon_like | http_slow_like>`

## timestamp

`<UTC timestamp>`

## Kali VM state

- VM name: `lab-attacker-kali`
- Hostname: `<observed hostname>`
- Host-only IP: `192.168.56.103`
- Readiness status: `<ready | warning | blocked>`

## target node

- Node ID: `<iot-drone-sitl | iot-smart-watch-medium>`
- Node IP: `<192.168.56.101 | 192.168.56.102>`
- Assigned tier: `<weak | medium>`

## observation method

`<controlled PacketWindow label, replay design, or future approved observation method>`

## PacketWindow summary

- Window size: `30`
- Scenario label: `<scenario_id>`
- Protocol tendency: `<ICMP | TCP SYN | HTTP/TCP>`
- Packet count: `<count>`
- Timing summary: `<summary>`

## extracted features

List the relevant 28-feature values or unsupported fields. Do not infer missing values silently.

## scaled mode

`<selected_12_scaled | original_28_scaled>`

## MQTT flow topic

`ids/flows/{node_id}`

## prediction topic

`ids/predictions/{node_id}`

## alert topic

`ids/alerts/{node_id}`

## dashboard evidence

- `/demo` screenshot reference: `<path or note>`
- Latest alert panel: `<visible | not visible>`
- Live event stream: `<visible | not visible>`
- Metrics panel: `<flows/predictions/alerts>`

## errors

- `final_ids_api_prediction_errors_total`: `<value>`
- `final_mqtt_bridge_prediction_errors_total`: `<value>`

## interpretation

Explain what the live deployment evidence shows. Keep scientific performance claims tied to P12/P13.

## limits

Document approximation, unsupported features, runtime warnings, and safety boundaries.

