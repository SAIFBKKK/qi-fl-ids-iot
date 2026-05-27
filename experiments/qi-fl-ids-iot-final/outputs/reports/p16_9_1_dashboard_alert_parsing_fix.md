# P16.9.1 Dashboard Alert Parsing Fix

## Objective

Fix the P16.9 dashboard alert presentation so IDS alerts show the real label and confidence instead of fallback values such as `label unknown` and `confidence n/a`.

## Root Cause

The dashboard was reading recent alerts through `online-validator /summary`. The validator exposed `payload_preview`, which is a truncated text preview. When an alert payload was truncated or not parsed, the dashboard could still see the MQTT topic but could not reliably recover:

- `predicted_label`
- `predicted_label_id`
- `confidence`

## Fix

- `online-validator` now stores the parsed JSON payload in recent samples for alerts, predictions, and status messages.
- The dashboard prefers `sample.payload` when present.
- The dashboard keeps compatibility with older summaries by falling back to `payload_preview` parsing.
- Alert label and confidence extraction now use robust field fallbacks:
  - label: `predicted_label`, `label`, `prediction_label`
  - confidence: `confidence`, `probability_attack`, `attack_probability`, `score`

## Expected Result

The dashboard should display:

```text
Alert detected on iot-smart-watch-medium
severity critical
label attack
confidence 0.990
flow p16-7-window-...
```

## Limit

If the running `online-validator` container has not been rebuilt, old samples may still contain only `payload_preview`. Rebuild/restart `online-validator` and `dashboard-p13` before final screenshots.
