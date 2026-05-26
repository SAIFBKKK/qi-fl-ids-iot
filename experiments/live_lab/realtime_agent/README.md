# Realtime Agent

P16.1 step 0 prepares the future real-time packet-window path but does not start
capture.

Prepared modules:

- `packet_capture.py`: dry-run capture plan only.
- `flow_window.py`: flow window skeleton.
- `feature_extractor.py`: placeholder 28-feature extraction with unsupported
  fields represented as `None`.
- `scaler_runtime.py`: future scaler runtime placeholder.
- `edge_inference.py`: future edge inference placeholder for the medium node.

Future path:

`packet window -> 28 features -> scaler -> selected QGA features -> IDS`

