# P16.11 Controlled Scenario Observation Design

## Objective

P16.11 defines how future controlled Kali scenario observations will be named, linked to packet windows, mapped to expected features, and verified through the live MQTT-to-IDS dashboard path.

This phase is design only. It does not activate Kali scenarios.

## Architecture

- Server Windows host: `192.168.56.1`
- VM1 `iot-rpi-weak`: `192.168.56.101`
- VM2 `iot-smart-watch-medium`: `192.168.56.102`
- VM3 `lab-attacker-kali`: `192.168.56.103`
- Dashboard demo: `http://192.168.56.1:8013/demo`
- Controller: `http://192.168.56.1:8020`
- final IDS API: `http://192.168.56.1:8014`
- final MQTT bridge: `http://192.168.56.1:8016`
- online-validator: `http://192.168.56.1:8015`
- MQTT broker: `192.168.56.1:1883`

## Kali Role

`lab-attacker-kali` is a future controlled scenario workstation. In P16.11 it only provides scenario labels and observation design context. It is not used to execute traffic, capture packets, or run tools.

## IoT Node Role

The IoT nodes remain the observable deployment targets:

- `iot-rpi-weak`: weak tier, `selected_12_scaled` path.
- `iot-smart-watch-medium`: medium tier, `original_28_scaled` path with QGA mask applied inside the API.

## PacketWindow(30) Role

Each future observation should be linked to a `PacketWindow(30)` summary:

- scenario label,
- target node,
- protocol tendency,
- feature vector mode,
- MQTT flow ID,
- extraction warnings if any.

## Scaler JSON Role

The runtime scaler JSON remains the deployable scaling artifact for VM-side feature preparation. The observation design must state whether the flow uses `selected_12_scaled` or `original_28_scaled`.

## final-mqtt-bridge Role

The bridge remains responsible for consuming `ids/flows/{node_id}`, calling final-ids-api, and publishing `ids/predictions/{node_id}` plus `ids/alerts/{node_id}`.

## final-ids-api Role

The API remains the final model endpoint for `p8_fedavg_qga_l1` and `selected_mask_id=conservative_seed_42`. For `original_28_scaled`, the API applies the QGA mask before inference.

## Dashboard /demo Role

The `/demo` page should show:

- connected devices,
- assigned tiers,
- model assignment,
- latest PacketWindow flow evidence,
- prediction and alert evidence,
- zero runtime errors.

## Target Flow

`Kali scenario label -> observation window -> features -> MQTT -> IDS -> dashboard alert`

In P16.11, this flow is a design target only.

## Limits

- No scenario activation.
- No scan.
- No flood.
- No live capture.
- No traffic generation.
- No scientific result changes.
- Scientific model evaluation remains P12/P13.
