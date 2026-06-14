# Dashboard Observation Flow

The `/demo` dashboard should present future controlled scenario evidence as operational deployment proof.

## Expected Presentation

- Scenario label visible.
- Target node visible.
- Latest PacketWindow(30) flow visible.
- Input mode visible.
- Prediction visible.
- Alert visible.
- Runtime errors visible and equal to zero.

## Evidence Path

1. A future approved scenario label is attached to an observation window.
2. The target IoT node produces or receives a PacketWindow(30) summary.
3. The feature extractor maps the window to the 28-feature schema.
4. The runtime scaler JSON prepares the selected mode.
5. The MQTT payload is published to `ids/flows/{node_id}`.
6. final-mqtt-bridge calls final-ids-api.
7. Predictions and alerts are published.
8. `/demo` displays the live event path and zero-error status.

## Dashboard Limits

The dashboard validates operational flow. It does not replace P12/P13 scientific evaluation.
