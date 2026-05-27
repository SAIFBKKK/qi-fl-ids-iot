# P16.6 Live MQTT Evidence Table

- Generated at: `2026-05-27T02:11:16.400021Z`
- Server URL: `http://192.168.56.1`
- Scope: controlled JSON flow payloads only; no packet capture and no offensive scenario.

## Node Evidence

| Node | Input mode | Manual flows | Live flows | Manual predictions | Live predictions | Manual alerts | Live alerts |
|---|---|---:|---:|---:|---:|---:|---:|
| `iot-rpi-weak` | `selected_12_scaled` | 1 | 1 | 1 | 1 | 1 | 1 |
| `iot-smart-watch-medium` | `original_28_scaled` | 1 | 1 | 1 | 1 | 1 | 1 |

## Error Counters

| Metric | Manual observed | Live extracted |
|---|---:|---:|
| `final_ids_api_prediction_errors_total` | 0 | 0 |
| `final_mqtt_bridge_prediction_errors_total` | 0 | 0 |

## Warnings

- None.