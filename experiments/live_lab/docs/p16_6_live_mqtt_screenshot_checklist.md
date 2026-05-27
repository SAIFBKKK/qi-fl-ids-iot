# P16.6 Live MQTT Screenshot Checklist

Capture these screenshots for the final live lab evidence package.

## Server Runtime

- `docker compose ps` showing the required services up.
- `http://192.168.56.1:8020/nodes` with `iot-rpi-weak` and `iot-smart-watch-medium`.
- `http://192.168.56.1:8020/assignments` with `iot-rpi-weak` as `weak` and `iot-smart-watch-medium` as `medium`.

## MQTT Observability

- A terminal observing `ids/flows/iot-rpi-weak`, `ids/predictions/iot-rpi-weak`, and `ids/alerts/iot-rpi-weak`.
- A terminal observing `ids/flows/iot-smart-watch-medium`, `ids/predictions/iot-smart-watch-medium`, and `ids/alerts/iot-smart-watch-medium`.

## Metrics

- `http://192.168.56.1:8016/metrics` showing flow, prediction, and alert counters for both VM nodes.
- `http://192.168.56.1:8015/summary` showing one flow, one prediction, and one alert topic for both VM nodes.
- `http://192.168.56.1:8014/metrics` showing prediction counters and `final_ids_api_prediction_errors_total` equal to zero.

## Dashboard

- Grafana runtime dashboard if available.
- dashboard-p13 live status if available.

## Notes

- The screenshots document a technical pipeline validation using controlled JSON flow payloads.
- They do not replace the scientific model evaluation from P12/P13.
- No packet capture or lab scenario is required for P16.6 screenshots.
